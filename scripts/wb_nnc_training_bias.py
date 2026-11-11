#!/usr/bin/env python3
"""SCRUM-6482 Task 1: measure how the WB training sets sample the corpus, per NNC bucket.

WB curators validate papers the in-house NNC classifier flagged HIGH far more often than the rest, so the ABC
training sets over-represent NNC-HIGH and under-represent NNC-NEG. For each topic this script compares, per NNC
bucket (HIGH / MEDIUM / LOW / NEG):

  - corpus: papers NNC assigned to the bucket (WB's nnc_<datatype> topic entity tags in the ABC);
  - training: those papers in the ABC training set for the topic, with their positive/negative split;
  - sampled %: training / corpus;
  - corpus mix % vs training mix %: the bucket's share of each;
  - weight: corpus mix / training mix, i.e. the per-paper weight that makes the training mix match the corpus
    (the "resample weighted by 1 / validation rate" variant of Task 2).

Only papers published on or after --date-from are counted, matching the trainer's default --filter_date_before
(papers without a readable year are kept, as the trainer keeps unparseable dates).

Reads the ABC database directly (read-only): PSQL_HOST, PSQL_PORT, PSQL_DATABASE, PSQL_USERNAME, PSQL_PASSWORD.

Usage:
    python scripts/wb_nnc_training_bias.py                                  # both pilot topics, latest datasets
    python scripts/wb_nnc_training_bias.py --topic ATP:0000061=nnc_catalyticact --papers-csv papers.csv
"""
import argparse
import csv
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import psycopg2

logger = logging.getLogger(__name__)

BUCKETS = ("HIGH", "MEDIUM", "LOW", "NEG")
NO_BUCKET = "no NNC"

# pilot topics: ATP topic -> WB NNC tag source method
PILOT_TOPICS = {
    "ATP:0000041": "nnc_otherexpr",      # expression
    "ATP:0000061": "nnc_catalyticact",   # catalytic activity
}

# One row per paper that has an NNC bucket or is in the training set (or both).
_SQL_PAPERS = """
WITH ds AS (
    SELECT d.id
    FROM dataset d JOIN mod m ON m.id = d.mod_id
    WHERE m.abbreviation = %(mod)s AND d.data_type = %(topic)s AND d.dataset_type = 'document'
      AND (%(version)s IS NULL OR d.version = %(version)s)
    ORDER BY d.version DESC
    LIMIT 1
),
nnc AS (
    SELECT t.reference_id, t.confidence_level AS bucket
    FROM topic_entity_tag t
    JOIN tag_source ts ON ts.id = t.tag_source_id
    JOIN mod m ON m.id = ts.secondary_data_provider_id
    WHERE m.abbreviation = %(mod)s AND ts.source_method = %(source_method)s AND t.topic = %(topic)s
),
training AS (
    SELECT r.id AS reference_id, e.classification_value AS cls
    FROM dataset_entry e JOIN reference r ON r.curie = e.reference_curie
    WHERE e.dataset_id = (SELECT id FROM ds) AND e.set_type = 'training'
)
SELECT r.curie, nnc.bucket, training.cls, substring(r.date_published from '^[0-9]{4}')
FROM nnc FULL JOIN training ON training.reference_id = nnc.reference_id
JOIN reference r ON r.id = coalesce(nnc.reference_id, training.reference_id)
WHERE NOT %(classified_only)s
   OR (r.category = 'Research_Article'
       AND EXISTS (SELECT 1 FROM reference_mod_referencetype rmr
                   JOIN mod_referencetype mrt ON mrt.id = rmr.mod_referencetype_id
                   JOIN referencetype rt ON rt.id = mrt.referencetype_id
                   JOIN mod m2 ON m2.id = mrt.mod_id
                   WHERE rmr.reference_id = r.id AND m2.abbreviation = %(mod)s AND rt.label = 'Experimental'))
"""

_SQL_DATASET = """
SELECT d.id, d.version
FROM dataset d JOIN mod m ON m.id = d.mod_id
WHERE m.abbreviation = %(mod)s AND d.data_type = %(topic)s AND d.dataset_type = 'document'
  AND (%(version)s IS NULL OR d.version = %(version)s)
ORDER BY d.version DESC
LIMIT 1
"""


@dataclass(frozen=True)
class Paper:
    curie: str
    bucket: str                  # HIGH / MEDIUM / LOW / NEG, or NO_BUCKET
    classification: Optional[str]  # positive / negative, or None if not in the training set
    year: Optional[int]


@dataclass(frozen=True)
class BucketStats:
    bucket: str
    corpus: int
    training: int
    positive: int
    negative: int
    sampled_pct: Optional[float]
    corpus_mix_pct: Optional[float]
    training_mix_pct: Optional[float]
    positive_rate_pct: Optional[float]
    weight: Optional[float]


def _pct(part: float, whole: float) -> Optional[float]:
    return round(100.0 * part / whole, 1) if whole else None


def bucket_stats(papers: Iterable[Paper], year_from: Optional[int] = None) -> List[BucketStats]:
    """Per-bucket corpus/training counts and mixes for the papers published in or after year_from (papers with
    no year are kept). Papers with no NNC bucket count only towards training."""
    corpus: Counter = Counter()
    training: Counter = Counter()
    positive: Counter = Counter()
    negative: Counter = Counter()
    for paper in papers:
        if year_from is not None and paper.year is not None and paper.year < year_from:
            continue
        if paper.bucket != NO_BUCKET:
            corpus[paper.bucket] += 1
        if paper.classification is not None:
            training[paper.bucket] += 1
            if paper.classification == "positive":
                positive[paper.bucket] += 1
            elif paper.classification == "negative":
                negative[paper.bucket] += 1

    corpus_total = sum(corpus[b] for b in BUCKETS)
    training_total = sum(training[b] for b in BUCKETS)
    rows = []
    for bucket in BUCKETS + (NO_BUCKET,):
        if bucket == NO_BUCKET and not training[bucket]:
            continue
        corpus_mix = _pct(corpus[bucket], corpus_total) if bucket != NO_BUCKET else None
        training_mix = _pct(training[bucket], training_total) if bucket != NO_BUCKET else None
        weight = None
        if bucket != NO_BUCKET and training[bucket] and corpus_total and training_total:
            weight = round((corpus[bucket] / corpus_total) / (training[bucket] / training_total), 2)
        rows.append(BucketStats(
            bucket=bucket, corpus=corpus[bucket], training=training[bucket],
            positive=positive[bucket], negative=negative[bucket],
            sampled_pct=_pct(training[bucket], corpus[bucket]) if bucket != NO_BUCKET else None,
            corpus_mix_pct=corpus_mix, training_mix_pct=training_mix,
            positive_rate_pct=_pct(positive[bucket], training[bucket]), weight=weight))
    return rows


def subsample_size(stats: List[BucketStats]) -> Tuple[int, Dict[str, int]]:
    """Largest training set whose bucket mix matches the corpus by dropping papers only (stratified
    subsampling): the scarcest bucket relative to its corpus share sets the size."""
    shares = {s.bucket: (s.training, s.corpus_mix_pct / 100.0) for s in stats
              if s.bucket != NO_BUCKET and s.corpus_mix_pct}
    if not shares:
        return 0, {}
    total = min(training / share for training, share in shares.values())
    per_bucket = {bucket: int(total * share) for bucket, (_, share) in shares.items()}
    return sum(per_bucket.values()), per_bucket


def fetch_papers(connection, mod: str, topic: str, source_method: str, version: Optional[int],
                 classified_only: bool = True) -> Tuple[Optional[Tuple[int, int]], List[Paper]]:
    params = {"mod": mod, "topic": topic, "source_method": source_method, "version": version,
              "classified_only": classified_only}
    with connection.cursor() as cursor:
        cursor.execute(_SQL_DATASET, params)
        dataset = cursor.fetchone()
        cursor.execute(_SQL_PAPERS, params)
        papers = [Paper(curie=curie, bucket=bucket or NO_BUCKET, classification=cls,
                        year=int(year) if year else None)
                  for curie, bucket, cls, year in cursor.fetchall()]
    return dataset, papers


def format_table(topic: str, dataset: Optional[Tuple[int, int]], stats: List[BucketStats]) -> str:
    def cell(value):
        return "" if value is None else f"{value:,}" if isinstance(value, int) else f"{value}"

    header = ("bucket", "corpus", "training", "positive", "negative", "sampled %", "corpus mix %",
              "training mix %", "positive rate %", "weight")
    lines = [f"## {topic}" + (f" (dataset {dataset[0]}, version {dataset[1]})" if dataset else " (no dataset)"),
             "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for s in stats:
        lines.append("| " + " | ".join(cell(v) for v in (
            s.bucket, s.corpus, s.training, s.positive, s.negative, s.sampled_pct, s.corpus_mix_pct,
            s.training_mix_pct, s.positive_rate_pct, s.weight)) + " |")
    size, per_bucket = subsample_size(stats)
    training_total = sum(s.training for s in stats if s.bucket != NO_BUCKET)
    lines += ["", f"Stratified subsampling to the corpus mix keeps {size:,} of {training_total:,} papers: "
              + ", ".join(f"{b} {n:,}" for b, n in per_bucket.items())]
    return "\n".join(lines)


def _parse_topic(value: str) -> Tuple[str, str]:
    topic, sep, source_method = value.partition("=")
    if not sep or not topic or not source_method:
        raise argparse.ArgumentTypeError(f"expected ATP_TOPIC=nnc_source_method, got {value!r}")
    return topic, source_method


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mod", default="WB")
    parser.add_argument("--topic", action="append", type=_parse_topic,
                        help="ATP_TOPIC=nnc_source_method; repeatable (default: the two pilot topics)")
    parser.add_argument("--dataset-version", type=int, help="dataset version (default: latest)")
    parser.add_argument("--date-from", default="2005-01-01",
                        help="count papers published in or after this year (YYYY or YYYY-MM-DD; '' for all)")
    parser.add_argument("--papers-csv", help="also write every paper (curie, topic, bucket, classification, year)")
    parser.add_argument("--all-papers", action="store_true",
                        help="count every paper; by default only the papers the ABC sends to classification "
                             "(research articles of reference type Experimental for the MOD) are counted")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    topics = args.topic or list(PILOT_TOPICS.items())
    year_from = int(args.date_from[:4]) if args.date_from else None
    connection = psycopg2.connect(
        host=os.environ["PSQL_HOST"], port=os.environ.get("PSQL_PORT", "5432"),
        dbname=os.environ["PSQL_DATABASE"], user=os.environ["PSQL_USERNAME"],
        password=os.environ["PSQL_PASSWORD"])
    connection.set_session(readonly=True)
    writer = None
    papers_file = open(args.papers_csv, "w", newline="") if args.papers_csv else None
    try:
        if papers_file:
            writer = csv.writer(papers_file)
            writer.writerow(["reference_curie", "topic", "nnc_bucket", "classification", "year"])
        for topic, source_method in topics:
            dataset, papers = fetch_papers(connection, args.mod, topic, source_method, args.dataset_version,
                                           classified_only=not args.all_papers)
            logger.info(f"{topic}: {len(papers):,} papers ({source_method}, dataset {dataset})")
            print(format_table(topic, dataset, bucket_stats(papers, year_from)))
            print()
            if writer:
                for paper in papers:
                    writer.writerow([paper.curie, topic, paper.bucket, paper.classification or "", paper.year or ""])
    finally:
        connection.close()
        if papers_file:
            papers_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
