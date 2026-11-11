#!/usr/bin/env python3
"""SCRUM-6482: draw the papers an LLM reads to estimate how common each WB data type is in the corpus.

Two sets, written as TSV files to OUT_DIR:

  random_sample.tsv       N papers drawn uniformly at random from the WB papers the ABC sends to topic
                          classification: WB corpus (mod_corpus_association.corpus = true), category
                          Research_Article and WB reference type Experimental (the workflow's
                          proceed_on_value::reference_type::Experimental gate), published on/after --date-from
                          (unreadable dates kept, as the trainer does), with a final converted_merged_main Markdown.
                          Papers in a WB document dataset are not excluded (the estimate is for the whole
                          classified corpus); in_dataset marks them.
  calibration_sample.tsv  per data type, up to --per-class curator-labelled positives and negatives from the
                          latest WB document dataset (any set type), with Markdown and on/after --date-from. Used
                          to measure the LLM's error rates against curator labels.

Also writes corpus_counts.tsv (corpus size with and without Markdown) so the estimate can say what share of the
corpus the sample represents.

Reads the ABC database read-only: PSQL_HOST, PSQL_PORT, PSQL_DATABASE, PSQL_USERNAME, PSQL_PASSWORD.

Usage:
    python scripts/wb_llm_prevalence_sample.py OUT_DIR [--n 1000] [--per-class 10] [--seed 6482] [--extend]

--extend keeps the eligible papers of an existing random_sample.tsv (a random sample filtered to the eligible papers
is still a uniform random sample of them), adds new eligible papers up to N, and keeps calibration_sample.tsv.
"""
import argparse
import csv
import os
import random
import re
from typing import Dict, List, Optional, Set, Tuple

import psycopg2

TOPICS = ["ATP:0000041", "ATP:0000061", "ATP:0000062", "ATP:0000068", "ATP:0000069", "ATP:0000070",
          "ATP:0000082", "ATP:0000083", "ATP:0000084", "ATP:0000096", "ATP:0000152"]
YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

CORPUS_SQL = """
SELECT r.id, r.curie, COALESCE(NULLIF(r.date_published, ''), NULLIF(r.date_published_start, '')) AS dt,
       EXISTS (SELECT 1 FROM referencefile rf WHERE rf.reference_id = r.id
               AND rf.file_class = 'converted_merged_main' AND rf.file_extension = 'md'
               AND rf.file_publication_status = 'final') AS has_md
FROM reference r
JOIN mod_corpus_association mca ON mca.reference_id = r.id
JOIN mod m ON m.id = mca.mod_id
WHERE m.abbreviation = 'WB' AND mca.corpus IS TRUE AND r.category = 'Research_Article'
  AND EXISTS (SELECT 1 FROM reference_mod_referencetype rmr
              JOIN mod_referencetype mrt ON mrt.id = rmr.mod_referencetype_id
              JOIN referencetype rt ON rt.id = mrt.referencetype_id
              WHERE rmr.reference_id = r.id AND mrt.mod_id = m.id AND rt.label = 'Experimental')
"""

DATASET_SQL = """
SELECT DISTINCT ON (d.data_type) d.id, d.data_type
FROM dataset d JOIN mod m ON m.id = d.mod_id
WHERE m.abbreviation = 'WB' AND d.dataset_type = 'document' AND d.data_type = ANY(%s)
ORDER BY d.data_type, d.version DESC
"""


def year_of(date_str: Optional[str]) -> Optional[int]:
    match = YEAR_RE.search(date_str or "")
    return int(match.group(1)) if match else None


def keep_by_date(date_str: Optional[str], year_from: int) -> bool:
    year = year_of(date_str)
    return year is None or year >= year_from


def connect():
    conn = psycopg2.connect(host=os.environ["PSQL_HOST"], port=os.environ.get("PSQL_PORT", "5432"),
                            dbname=os.environ["PSQL_DATABASE"], user=os.environ["PSQL_USERNAME"],
                            password=os.environ["PSQL_PASSWORD"])
    conn.set_session(readonly=True)
    return conn


def load_datasets(cur) -> Tuple[Set[str], Dict[str, List[Tuple[str, str]]]]:
    """All curies in the latest WB document datasets, and per topic its (curie, label) entries."""
    cur.execute(DATASET_SQL, (TOPICS,))
    datasets = cur.fetchall()
    all_curies: Set[str] = set()
    labelled: Dict[str, List[Tuple[str, str]]] = {}
    for dataset_id, topic in datasets:
        cur.execute("SELECT DISTINCT reference_curie, classification_value FROM dataset_entry WHERE dataset_id = %s",
                    (dataset_id,))
        rows = cur.fetchall()
        labelled[topic] = [(curie, value) for curie, value in rows]
        all_curies.update(curie for curie, _ in rows)
    return all_curies, labelled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("--n", type=int, default=1000, help="random sample size (default 1000)")
    parser.add_argument("--per-class", type=int, default=10,
                        help="calibration papers per data type and class (default 10)")
    parser.add_argument("--date-from", type=int, default=2005, help="earliest publication year (default 2005)")
    parser.add_argument("--seed", type=int, default=6482)
    parser.add_argument("--extend", action="store_true",
                        help="keep the eligible papers of the existing samples and top the random one up to --n")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(CORPUS_SQL)
        corpus = {curie: (ref_id, dt, has_md) for ref_id, curie, dt, has_md in cur.fetchall()}
        dataset_curies, labelled = load_datasets(cur)

    in_window = {c: v for c, v in corpus.items() if keep_by_date(v[1], args.date_from)}
    eligible = sorted(c for c, (_, _, has_md) in in_window.items() if has_md)
    random_path = os.path.join(args.out_dir, "random_sample.tsv")
    kept: List[str] = []
    if args.extend and os.path.exists(random_path):
        with open(random_path) as fh:
            kept = [row["curie"] for row in csv.DictReader(fh, delimiter="\t") if row["curie"] in set(eligible)]
    pool = [c for c in eligible if c not in set(kept)]
    random_curies = kept + rng.sample(pool, min(args.n - len(kept), len(pool)))

    with open(os.path.join(args.out_dir, "corpus_counts.tsv"), "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["what", "papers"])
        writer.writerow(["wb_corpus", len(corpus)])
        writer.writerow([f"wb_corpus_from_{args.date_from}", len(in_window)])
        writer.writerow([f"wb_corpus_from_{args.date_from}_with_md", sum(1 for v in in_window.values() if v[2])])
        writer.writerow(["in_wb_document_datasets", len(dataset_curies & set(in_window))])
        writer.writerow(["eligible_for_random_sample", len(eligible)])
        writer.writerow(["random_sample", len(random_curies)])
        writer.writerow(["random_sample_kept_from_previous", len(kept)])

    with open(random_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["curie", "date", "in_dataset"])
        for curie in random_curies:
            writer.writerow([curie, in_window[curie][1] or "", int(curie in dataset_curies)])

    calibration_path = os.path.join(args.out_dir, "calibration_sample.tsv")
    if args.extend and os.path.exists(calibration_path):
        print(open(os.path.join(args.out_dir, "corpus_counts.tsv")).read())
        return
    taken = set(random_curies)
    with open(calibration_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["curie", "topic", "curator_label"])
        for topic in TOPICS:
            for label in ("positive", "negative"):
                pool = sorted(c for c, v in labelled.get(topic, []) if v == label and c in in_window
                              and in_window[c][2] and c not in taken)
                for curie in rng.sample(pool, min(args.per_class, len(pool))):
                    writer.writerow([curie, topic, label])
    print(open(os.path.join(args.out_dir, "corpus_counts.tsv")).read())


if __name__ == "__main__":
    main()
