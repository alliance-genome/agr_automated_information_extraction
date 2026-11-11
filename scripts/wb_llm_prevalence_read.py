#!/usr/bin/env python3
"""SCRUM-6482: have Claude read each sampled WB paper and decide, for each of the 11 WB data types, whether it has
curatable data of that type.

Uses headless Claude Code (`claude -p`), so it runs on the user's Claude subscription: one call per paper, no tools,
the same instructions for every paper (docs/wb_datatype_definitions/, without their evidence sections), and an answer
validated against a JSON schema. Results go to OUT_DIR/results/<model>/<curie>.json, one file per paper and model, so
the run can stop (at a usage limit, or at --stop-at) and resume: papers that already have a result are skipped.
Keeping each model's answers apart lets the same papers be read by two models and compared.

Usage:
    python scripts/wb_llm_prevalence_read.py MD_DIR OUT_DIR CURIE_FILE [CURIE_FILE ...] [--workers 4]
        [--model claude-opus-5-5] [--max-chars 400000] [--limit N] [--stop-at HH:MM]

CURIE_FILE: a TSV with a "curie" column (random_sample.tsv, calibration_sample.tsv), or a list of curies.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional

DEFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "wb_datatype_definitions")
TOPICS = ["ATP:0000041", "ATP:0000061", "ATP:0000062", "ATP:0000068", "ATP:0000069", "ATP:0000070",
          "ATP:0000082", "ATP:0000083", "ATP:0000084", "ATP:0000096", "ATP:0000152"]
MIN_CHARS = 1500
REFERENCES_RE = re.compile(r"^#{1,4}\s*(\d+\.?\s*)?(references|bibliography|literature cited|works cited|"
                           r"reference list)\s*$", re.I | re.M)

ANSWER_INSTRUCTIONS = """
# Answer

Answer under the key of each data type, written with an underscore (ATP:0000041 -> `ATP_0000041`).
For every data type give `positive` (true or false) and `evidence`: for a positive, a short quote or pointer from the
paper (figure, table, section) showing the data; for a negative, a few words on why (e.g. "review", "only cited",
"mammalian cells only", "no RNAi"). Keep each evidence under 200 characters. Also give `paper_kind`.
The paper text may have been cut at the end if it was very long; judge from what you have.
"""


def build_system_prompt() -> str:
    parts = [open(os.path.join(DEFS_DIR, "prompt_header.md")).read(),
             open(os.path.join(DEFS_DIR, "common_rules.md")).read(), "# Data types\n"]
    for path in sorted(glob.glob(os.path.join(DEFS_DIR, "ATP_*.md"))):
        text = open(path).read().split("\n## Evidence")[0]
        text = re.sub(r"\*\*Status:\*\*[^\n]*\n", "", text)
        parts.append(text.replace("# ATP:", "## ATP:", 1).strip() + "\n")
    parts.append(ANSWER_INSTRUCTIONS)
    return "\n".join(parts)


def build_schema() -> Dict:
    item = {"type": "object", "properties": {"positive": {"type": "boolean"}, "evidence": {"type": "string"}},
            "required": ["positive", "evidence"], "additionalProperties": False}
    keys = [t.replace(":", "_") for t in TOPICS]
    props = {k: item for k in keys}
    props["paper_kind"] = {"type": "string", "enum": ["primary research", "review", "methods or protocol",
                                                      "database or resource", "erratum or correction", "other"]}
    return {"type": "object", "properties": props, "required": keys + ["paper_kind"], "additionalProperties": False}


def strip_references(text: str) -> str:
    """Drop the reference list (from the last references heading in the second half of the paper, else the first
    one past 30% of it): it does not help the decision and is about a fifth of the text."""
    matches = list(REFERENCES_RE.finditer(text))
    if not matches:
        return text
    match = matches[-1] if matches[-1].start() > len(text) * 0.4 else matches[0]
    return text[:match.start()] if match.start() > len(text) * 0.3 else text


def find_claude() -> str:
    """Path of the claude executable. An auto-update can remove it for a few seconds, so wait for it."""
    for _ in range(20):
        path = shutil.which("claude") or next(
            (p for p in (os.path.expanduser("~/.local/bin/claude"), os.path.expanduser("~/.claude/local/claude"))
             if os.path.exists(p)), None)
        if path:
            return path
        time.sleep(15)
    raise FileNotFoundError("claude executable not found")


def stop_deadline(stop_at: Optional[str]) -> Optional[datetime]:
    """The next time the local clock shows HH:MM (today if still ahead, else tomorrow), or None."""
    if not stop_at:
        return None
    hour, minute = (int(x) for x in stop_at.split(":"))
    now = datetime.now()
    deadline = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return deadline if deadline > now else deadline + timedelta(days=1)


def read_curies(paths: List[str]) -> List[str]:
    curies: List[str] = []
    for path in paths:
        with open(path) as fh:
            first = fh.readline()
            fh.seek(0)
            if "curie" in first.split("\t"):
                curies += [row["curie"] for row in csv.DictReader(fh, delimiter="\t")]
            else:
                curies += [line.strip() for line in fh if line.startswith("AGRKB:")]
    return list(dict.fromkeys(curies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("md_dir")
    parser.add_argument("out_dir")
    parser.add_argument("curie_files", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max-chars", type=int, default=400000,
                        help="cut papers longer than this many characters (default 400000, ~100k tokens)")
    parser.add_argument("--limit", type=int, default=None, help="read at most N new papers")
    parser.add_argument("--stop-at", default=None,
                        help="local time HH:MM after which no new paper is started (e.g. 07:00 for a night run)")
    args = parser.parse_args()

    results_dir = os.path.join(args.out_dir, "results", args.model)
    os.makedirs(results_dir, exist_ok=True)
    system_prompt = build_system_prompt()
    with open(os.path.join(args.out_dir, "system_prompt.md"), "w") as fh:
        fh.write(system_prompt)
    schema = json.dumps(build_schema())

    todo = [c for c in read_curies(args.curie_files)
            if not os.path.exists(os.path.join(results_dir, c.replace(":", "_") + ".json"))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} papers to read", flush=True)
    stop = threading.Event()
    lock = threading.Lock()
    totals = {"done": 0, "failed": 0, "cost_usd": 0.0}
    deadline = stop_deadline(args.stop_at)

    def read_one(curie: str) -> str:
        if stop.is_set() or (deadline and datetime.now() >= deadline):
            stop.set()
            return f"SKIP {curie}"
        md_path = os.path.join(args.md_dir, curie.replace(":", "_") + ".md")
        if not os.path.exists(md_path):
            return f"NO_MD {curie}"
        full = open(md_path, errors="replace").read()
        text = strip_references(full)
        record = {"curie": curie, "chars": len(full), "chars_sent": min(len(text), args.max_chars),
                  "truncated": len(text) > args.max_chars, "model": args.model}
        if len(text) < MIN_CHARS:
            record["status"] = "too_short"
        else:
            proc = subprocess.run(
                [find_claude(), "-p", "--model", args.model, "--effort", args.effort, "--tools", "",
                 "--no-session-persistence", "--output-format", "json", "--system-prompt", system_prompt,
                 "--json-schema", schema,
                 # no MCP servers: claude.ai connectors would add 100k+ tokens of tool definitions per paper and
                 # could let the model look things up instead of judging the paper text alone
                 "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}', "--setting-sources", ""],
                input=text[:args.max_chars], capture_output=True, text=True, timeout=1800,
                # each paper is read once, so caching it only adds the cache-write premium
                env={**os.environ, "DISABLE_PROMPT_CACHING": "1"})
            try:
                out = json.loads(proc.stdout)
            except json.JSONDecodeError:
                out = {"is_error": True, "result": (proc.stdout + proc.stderr)[-500:]}
            message = str(out.get("result", ""))[:300]
            if out.get("stop_reason") == "refusal" or re.search(r"safety classifier|won't call StructuredOutput",
                                                                message, re.I):
                record.update(status="refused", detail=message)
            elif out.get("is_error") or not out.get("structured_output"):
                if re.search(r"limit|429|overloaded|quota", message, re.I):
                    stop.set()
                with lock:
                    totals["failed"] += 1
                return f"FAIL {curie}: {message}"
            else:
                record.update(status="ok", answer=out["structured_output"], usage=out.get("usage"),
                              num_turns=out.get("num_turns"),
                              cost_usd_equivalent=out.get("total_cost_usd"))
        with open(os.path.join(results_dir, curie.replace(":", "_") + ".json"), "w") as fh:
            json.dump(record, fh)
        with lock:
            totals["done"] += 1
            totals["cost_usd"] += record.get("cost_usd_equivalent") or 0
        return f"OK {curie} {record['status']}"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(read_one, c) for c in todo]
        for i, future in enumerate(as_completed(futures), 1):
            line = future.result()
            if not line.startswith("OK") or i % 25 == 0:
                print(f"[{i}/{len(todo)}] {line} | done={totals['done']} failed={totals['failed']} "
                      f"cost_equiv=${totals['cost_usd']:.2f}", flush=True)
    if stop.is_set():
        print("Stopped early (usage limit or --stop-at). Re-run the same command later to resume.", flush=True)
        sys.exit(2)
    print(f"Finished: done={totals['done']} failed={totals['failed']} cost_equiv=${totals['cost_usd']:.2f}")


if __name__ == "__main__":
    main()
