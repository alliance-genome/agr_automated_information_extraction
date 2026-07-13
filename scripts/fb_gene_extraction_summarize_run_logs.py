#!/usr/bin/env python3
"""
Summarise ``RUN SUMMARY`` counters across a directory of run logs into a TSV.

Each log file is expected to contain a line of the form::

    2026-07-13 13:33:20 - __main__ - INFO - RUN SUMMARY: {'jobs_loaded': 0, ...}

emitted by ``bert_entity_extraction/annotation_helper.py``. This script pulls
that dict out of every ``*.log`` file in a directory and writes one row per
file to a tab-separated file for easy viewing.

Usage:
    python3 scripts/summarize_run_logs.py <logs_dir> [-o run_summary.tsv]

Columns: date, filename, then one column per counter (union across all files,
in first-seen order). Rows are sorted by date, then filename. Files without a
parseable RUN SUMMARY line are skipped with a warning on stderr.
"""

import argparse
import ast
import csv
import re
import sys
from pathlib import Path

# Captures the dict repr that follows "RUN SUMMARY:" on a log line.
RUN_SUMMARY_RE = re.compile(r"RUN SUMMARY:\s*(\{.*\})")
# Date embedded in the filename as _MMDDYYYY (e.g. run_..._07132026.log).
FILENAME_DATE_RE = re.compile(r"_(\d{2})(\d{2})(\d{4})")
# First "YYYY-MM-DD" timestamp on a log line, used as a fallback.
TIMESTAMP_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def parse_run_summary(text):
    """Return the counter dict from the last RUN SUMMARY line, or None."""
    matches = RUN_SUMMARY_RE.findall(text)
    if not matches:
        return None
    try:
        counters = ast.literal_eval(matches[-1])
    except (ValueError, SyntaxError):
        return None
    return counters if isinstance(counters, dict) else None


def extract_date(filename, text):
    """Derive an ISO date (YYYY-MM-DD) from the filename, else the log body."""
    match = FILENAME_DATE_RE.search(filename)
    if match:
        month, day, year = match.groups()
        return f"{year}-{month}-{day}"
    match = TIMESTAMP_DATE_RE.search(text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    return ""


def collect_rows(logs_dir):
    """Parse every *.log file in logs_dir into (row_dict, ordered_keys)."""
    rows = []
    counter_keys = []  # preserve first-seen order across files
    seen_keys = set()

    for log_path in sorted(logs_dir.glob("*.log")):
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"WARNING: could not read {log_path}: {exc}", file=sys.stderr)
            continue

        counters = parse_run_summary(text)
        if counters is None:
            print(f"WARNING: no RUN SUMMARY found in {log_path}, skipping",
                  file=sys.stderr)
            continue

        for key in counters:
            if key not in seen_keys:
                seen_keys.add(key)
                counter_keys.append(key)

        row = {"date": extract_date(log_path.name, text),
               "filename": log_path.name}
        row.update({str(k): v for k, v in counters.items()})
        rows.append(row)

    rows.sort(key=lambda r: (r["date"], r["filename"]))
    return rows, counter_keys


def write_tsv(rows, counter_keys, output_path):
    """Write rows to a tab-separated file with a header line."""
    fieldnames = ["date", "filename"] + counter_keys
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t",
                                restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs_dir", nargs="?", default="logs", type=Path,
                        help="Directory containing *.log files (default: logs)")
    parser.add_argument("-o", "--output", default="run_summary.tsv", type=Path,
                        help="Output TSV path (default: run_summary.tsv)")
    args = parser.parse_args()

    if not args.logs_dir.is_dir():
        parser.error(f"logs_dir is not a directory: {args.logs_dir}")

    rows, counter_keys = collect_rows(args.logs_dir)
    if not rows:
        print(f"No RUN SUMMARY lines found under {args.logs_dir}",
              file=sys.stderr)
        sys.exit(1)

    write_tsv(rows, counter_keys, args.output)
    print(f"Wrote {len(rows)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
