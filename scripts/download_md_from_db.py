#!/usr/bin/env python3
"""
Download converted_merged_main markdown files from the literature DB
(referencefile metadata) + the agr-literature S3 bucket.

Mirrors agr_literature_service download logic:
  referencefile.md5sum -> get_s3_folder_from_md5sum(ENV_STATE) -> <bucket>/<folder>/<md5sum>.gz
then gunzip to <out_dir>/AGRKB_<id>.md

Credentials are taken from the already-exported environment (source .env_cc first):
  PSQL_HOST/PORT/DATABASE/USERNAME/PASSWORD, AWS_ACCESS_KEY_ID/SECRET, ENV_STATE.

Two ways to choose which references to fetch:
  1. From a curie-list file (lines starting with "AGRKB:"):
       python3 scripts/download_md_from_db.py --curie-file LIST.txt OUT_DIR
  2. Every paper in a MOD corpus that has a converted_merged_main md (default WB):
       python3 scripts/download_md_from_db.py --all-mod WB OUT_DIR

Other flags:
  --threads N        parallel S3 downloads (default 24)
  --skip-existing    don't re-download a curie whose .md already exists in OUT_DIR
  --include-non-corpus   with --all-mod, include refs where mod_corpus_association.corpus
                         is false/null (default: corpus = true only, matching the pipeline)

Usage:
  set -a; source /Users/shuai/claude_code/agr_literature_service/.env_cc; set +a
  python3 scripts/download_md_from_db.py --all-mod WB ../abc/wb_all_md/
"""
import argparse
import gzip
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import psycopg2

BUCKET = "agr-literature"


def s3_folder_from_md5sum(md5sum: str) -> str:
    env_state = os.environ.get("ENV_STATE", "")
    if env_state in ("", "test"):
        folder = "test"
    elif env_state == "prod":
        folder = "prod"
    else:
        folder = "develop"
    folder += "/reference/documents/"
    folder += "/".join(list(md5sum[0:4]))
    return folder


def read_curies(path: str) -> list[str]:
    curies = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("AGRKB:"):
                curies.append(line)
    return curies


def _connect():
    return psycopg2.connect(
        host=os.environ["PSQL_HOST"], port=os.environ["PSQL_PORT"],
        dbname=os.environ["PSQL_DATABASE"], user=os.environ["PSQL_USERNAME"],
        password=os.environ["PSQL_PASSWORD"], connect_timeout=15,
    )


def md5sums_for_curies(curies: list[str]) -> dict:
    cn = _connect()
    cur = cn.cursor()
    cur.execute(
        """
        SELECT r.curie, rf.md5sum
        FROM reference r
        JOIN referencefile rf ON rf.reference_id = r.reference_id
        WHERE r.curie = ANY(%s)
          AND rf.file_class = 'converted_merged_main'
          AND rf.file_extension = 'md'
        """,
        (curies,),
    )
    rows = cur.fetchall()
    cn.close()
    return {c: m for c, m in rows}


def md5sums_for_mod(mod_abbr: str, corpus_only: bool) -> dict:
    cn = _connect()
    cur = cn.cursor()
    corpus_clause = "AND mca.corpus = true" if corpus_only else ""
    cur.execute(
        f"""
        SELECT DISTINCT ON (r.curie) r.curie, rf.md5sum
        FROM reference r
        JOIN mod_corpus_association mca ON mca.reference_id = r.reference_id
        JOIN mod m ON m.mod_id = mca.mod_id AND m.abbreviation = %s
        JOIN referencefile rf ON rf.reference_id = r.reference_id
         AND rf.file_class = 'converted_merged_main'
         AND rf.file_extension = 'md'
        {corpus_clause}
        ORDER BY r.curie, rf.referencefile_id
        """,
        (mod_abbr,),
    )
    rows = cur.fetchall()
    cn.close()
    return {c: m for c, m in rows}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--curie-file", help="file with AGRKB: curies, one per line")
    src.add_argument("--all-mod", metavar="MOD", help="all corpus papers for this MOD (e.g. WB)")
    ap.add_argument("out_dir")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--include-non-corpus", action="store_true",
                    help="with --all-mod, include refs not flagged corpus=true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.curie_file:
        curies = read_curies(args.curie_file)
        print(f"{len(curies)} curies from file")
        found = md5sums_for_curies(curies)
        missing = [c for c in curies if c not in found]
        print(f"DB returned md5sums for {len(found)}/{len(curies)} curies")
        if missing:
            print(f"NO md referencefile for {len(missing)}: {', '.join(missing[:20])}"
                  + (" ..." if len(missing) > 20 else ""))
    else:
        found = md5sums_for_mod(args.all_mod, corpus_only=not args.include_non_corpus)
        print(f"{args.all_mod}: {len(found)} papers with converted_merged_main md "
              f"({'corpus only' if not args.include_non_corpus else 'incl. non-corpus'})")

    items = sorted(found.items())
    if args.skip_existing:
        before = len(items)
        items = [(c, m) for c, m in items
                 if not os.path.exists(os.path.join(args.out_dir, f"AGRKB_{c.split(':', 1)[1]}.md"))]
        print(f"skip-existing: {before - len(items)} already present, {len(items)} to fetch")

    s3 = boto3.client("s3")
    counts = {"ok": 0, "fail": 0}
    lock = threading.Lock()

    def fetch(curie_md5):
        curie, md5sum = curie_md5
        folder = s3_folder_from_md5sum(md5sum)
        key = f"{folder}/{md5sum}.gz"
        out_path = os.path.join(args.out_dir, f"AGRKB_{curie.split(':', 1)[1]}.md")
        gz_path = out_path + ".gz"
        try:
            s3.download_file(BUCKET, key, gz_path)
            with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
                f_out.write(f_in.read())
            os.remove(gz_path)
            with lock:
                counts["ok"] += 1
            return None
        except Exception as e:
            if os.path.exists(gz_path):
                os.remove(gz_path)
            with lock:
                counts["fail"] += 1
            return f"FAIL {curie} (s3://{BUCKET}/{key}): {e}"

    total = len(items)
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(fetch, it): it for it in items}
        done = 0
        for fut in as_completed(futures):
            err = fut.result()
            if err:
                print("  " + err)
            done += 1
            if done % 2000 == 0:
                print(f"  ... {done}/{total} (ok={counts['ok']} fail={counts['fail']})")
    print(f"Downloaded {counts['ok']}, failed {counts['fail']} (of {total})")


if __name__ == "__main__":
    main()
