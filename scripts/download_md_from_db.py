#!/usr/bin/env python3
"""
Download converted_merged_main markdown files for a list of AGRKB reference curies,
using the literature DB (referencefile metadata) + the agr-literature S3 bucket.

Mirrors agr_literature_service download logic:
  referencefile.md5sum -> get_s3_folder_from_md5sum(ENV_STATE) -> <bucket>/<folder>/<md5sum>.gz
then gunzip to <out_dir>/AGRKB_<id>.md

Credentials are taken from the already-exported environment (source .env_cc first):
  PSQL_HOST/PORT/DATABASE/USERNAME/PASSWORD, AWS_ACCESS_KEY_ID/SECRET, ENV_STATE.

Usage:
  set -a; source /Users/shuai/claude_code/agr_literature_service/.env_cc; set +a
  python3 scripts/download_md_from_db.py CURIE_LIST_FILE OUT_DIR
"""
import gzip
import os
import sys

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


def main():
    curie_file, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    curies = read_curies(curie_file)
    print(f"{len(curies)} curies to fetch -> {out_dir}")

    cn = psycopg2.connect(
        host=os.environ["PSQL_HOST"], port=os.environ["PSQL_PORT"],
        dbname=os.environ["PSQL_DATABASE"], user=os.environ["PSQL_USERNAME"],
        password=os.environ["PSQL_PASSWORD"], connect_timeout=15,
    )
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
    found = {c: m for c, m in rows}
    print(f"DB returned md5sums for {len(found)}/{len(curies)} curies")
    missing = [c for c in curies if c not in found]
    if missing:
        print(f"NO md referencefile in DB for {len(missing)}: {', '.join(missing)}")

    s3 = boto3.client("s3")
    ok = fail = 0
    for curie, md5sum in sorted(found.items()):
        folder = s3_folder_from_md5sum(md5sum)
        key = f"{folder}/{md5sum}.gz"
        out_path = os.path.join(out_dir, f"AGRKB_{curie.split(':', 1)[1]}.md")
        gz_path = out_path + ".gz"
        try:
            s3.download_file(BUCKET, key, gz_path)
            with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
                f_out.write(f_in.read())
            os.remove(gz_path)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAIL {curie} (s3://{BUCKET}/{key}): {e}")
            if os.path.exists(gz_path):
                os.remove(gz_path)
    print(f"Downloaded {ok}, failed {fail}")


if __name__ == "__main__":
    main()
