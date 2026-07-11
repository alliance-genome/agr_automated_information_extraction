#!/usr/bin/env python3
"""
Download converted_merged_main markdown files for ZFIN test papers from the
literature DB (referencefile metadata) + the agr-literature S3 bucket.

This is a variant of scripts/download_md_from_db.py.  The WB script keys off
reference.curie (AGRKB curies).  ZFIN test papers are identified by their ZFIN
publication id (e.g. ZDB-PUB-241218-8), which lives in the cross_reference
table as curie "ZFIN:ZDB-PUB-241218-8".  So here we resolve reference_id via
cross_reference instead of reference.curie.

Mirrors agr_literature_service download logic:
  referencefile.md5sum -> get_s3_folder_from_md5sum(ENV_STATE) -> <bucket>/<folder>/<md5sum>.gz
then gunzip to <out_dir>/ZFIN_<id>.md

Credentials are taken from the already-exported environment (source .env_cc first):
  PSQL_HOST/PORT/DATABASE/USERNAME/PASSWORD, AWS_ACCESS_KEY_ID/SECRET, ENV_STATE.

Choose which references to fetch either from the ZFIN test TSV (first column is
the ZFIN pub id, one header line) or from a plain list of ids:
  1. From the ZFIN test-papers TSV:
       python3 scripts/download_md_from_db_zfin.py --tsv-file ../abc/ZFIN/ZFIN_test_papers.tsv OUT_DIR
  2. From a plain id/curie list (one per line; "ZFIN:" prefix optional):
       python3 scripts/download_md_from_db_zfin.py --id-file ids.txt OUT_DIR

Other flags:
  --prefix PREFIX    cross_reference curie prefix to prepend (default ZFIN)
  --threads N        parallel S3 downloads (default 24)
  --skip-existing    don't re-download a curie whose .md already exists in OUT_DIR

Usage:
  set -a; source /Users/shuai/claude_code/agr_literature_service/.env_cc; set +a
  python3 scripts/download_md_from_db_zfin.py --tsv-file ../abc/ZFIN/ZFIN_test_papers.tsv ../abc/ZFIN/zfin_test_md/
"""
import argparse
import csv
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


def _normalize_curie(raw: str, prefix: str) -> str:
    """Return "<prefix>:<id>"; accept ids that already carry the prefix."""
    raw = raw.strip()
    if not raw:
        return ""
    if ":" in raw and raw.split(":", 1)[0] == prefix:
        return raw
    return f"{prefix}:{raw}"


def read_ids_from_tsv(path: str, prefix: str, id_col: int = 0) -> list[str]:
    curies = []
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)  # skip header row
        for row in reader:
            if not row or len(row) <= id_col:
                continue
            curie = _normalize_curie(row[id_col], prefix)
            if curie:
                curies.append(curie)
    return curies


def read_ids_from_list(path: str, prefix: str) -> list[str]:
    curies = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            curie = _normalize_curie(line, prefix)
            if curie:
                curies.append(curie)
    return curies


def _connect():
    return psycopg2.connect(
        host=os.environ["PSQL_HOST"], port=os.environ["PSQL_PORT"],
        dbname=os.environ["PSQL_DATABASE"], user=os.environ["PSQL_USERNAME"],
        password=os.environ["PSQL_PASSWORD"], connect_timeout=15,
    )


def md5sums_for_xref_curies(curies: list[str]) -> dict:
    """Resolve reference_id via cross_reference (not reference.curie), then
    fetch the converted_merged_main md md5sum for each.  Returns {curie: md5sum}."""
    cn = _connect()
    cur = cn.cursor()
    cur.execute(
        """
        SELECT DISTINCT ON (xr.curie) xr.curie, rf.md5sum
        FROM cross_reference xr
        JOIN reference r ON r.reference_id = xr.reference_id
        JOIN referencefile rf ON rf.reference_id = r.reference_id
         AND rf.file_class = 'converted_merged_main'
         AND rf.file_extension = 'md'
        WHERE xr.curie = ANY(%s)
          AND xr.is_obsolete = false
        ORDER BY xr.curie, rf.referencefile_id
        """,
        (curies,),
    )
    rows = cur.fetchall()
    cn.close()
    return {c: m for c, m in rows}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tsv-file", help="ZFIN test-papers TSV; first column is the pub id")
    src.add_argument("--id-file", help="plain list of ids/curies, one per line")
    ap.add_argument("out_dir")
    ap.add_argument("--prefix", default="ZFIN", help="cross_reference curie prefix (default ZFIN)")
    ap.add_argument("--id-col", type=int, default=0, help="0-based column of the id in the TSV")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.tsv_file:
        curies = read_ids_from_tsv(args.tsv_file, args.prefix, args.id_col)
    else:
        curies = read_ids_from_list(args.id_file, args.prefix)
    print(f"{len(curies)} curies from input")

    found = md5sums_for_xref_curies(curies)
    missing = [c for c in curies if c not in found]
    print(f"DB returned md5sums for {len(found)}/{len(curies)} curies")
    if missing:
        print(f"NO md referencefile for {len(missing)}: {', '.join(missing[:20])}"
              + (" ..." if len(missing) > 20 else ""))

    items = sorted(found.items())
    if args.skip_existing:
        before = len(items)
        items = [(c, m) for c, m in items
                 if not os.path.exists(os.path.join(args.out_dir, f"{c.replace(':', '_')}.md"))]
        print(f"skip-existing: {before - len(items)} already present, {len(items)} to fetch")

    s3 = boto3.client("s3")
    counts = {"ok": 0, "fail": 0}
    lock = threading.Lock()

    def fetch(curie_md5):
        curie, md5sum = curie_md5
        folder = s3_folder_from_md5sum(md5sum)
        key = f"{folder}/{md5sum}.gz"
        out_path = os.path.join(args.out_dir, f"{curie.replace(':', '_')}.md")
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
