#!/usr/bin/env python3
"""SCRUM-6482: estimate how common each WB data type is in the corpus from the LLM reading of a random sample.

Inputs (OUT_DIR, as written by wb_llm_prevalence_sample.py and wb_llm_prevalence_read.py):
  random_sample.tsv, calibration_sample.tsv, results/<model>/<curie>.json

Error rates: for each data type, the LLM's answers on curator-labelled papers - the calibration sample plus
random-sample papers that are in that data type's WB document dataset - give sensitivity (curator positives the LLM
calls positive) and specificity (curator negatives the LLM calls negative).

Share of positives: the raw share of random-sample papers the LLM calls positive, and that share corrected for the
LLM's error rates (Rogan-Gladen: (raw + specificity - 1) / (sensitivity + specificity - 1), clipped to [0, 1]), with a
95% bootstrap interval that resamples both the random sample and the labelled papers. Papers the LLM could not read
("too_short") are left out.

Reads the ABC database read-only for the dataset labels: PSQL_HOST, PSQL_PORT, PSQL_DATABASE, PSQL_USERNAME,
PSQL_PASSWORD. Writes OUT_DIR/estimate_<model>.tsv and prints a table.

Per NNC bucket (--nnc_strata_csv, the per-paper CSV of scripts/wb_nnc_training_bias.py): the share of positives
among the random-sample papers in each NNC bucket of each topic, for the trainer's --rebalance
bucket_and_positive_mix and bucket_and_positive_mix_balanced. The raw
bucket share is shrunk towards the topic's overall raw share (as if --shrink extra papers at that share were
added; it steadies the small HIGH and MEDIUM buckets), then corrected with the topic's sensitivity and
specificity. Where the specificity is below --min_specificity the correction is not trusted (the curator
negatives it is measured on are hard cases, so it over-corrects) and the midpoint of the raw and corrected shares
is used instead. Shares are kept within [--floor, 1 - --floor] so no class in a bucket gets weight 0. Writes
OUT_DIR/bucket_prevalence_<model>.tsv (topic, bucket, papers, llm_positive, raw_share, shrunk_share,
corrected_share, positive_share, method); the trainer reads topic, bucket and positive_share.

Usage:
    python scripts/wb_llm_prevalence_estimate.py OUT_DIR [--model claude-opus-5-5] [--bootstrap 2000] [--seed 6482]
        [--nnc_strata_csv wb_nnc_papers.csv] [--shrink 10] [--min_specificity 0.9] [--floor 0.005]
"""
import argparse
import csv
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import psycopg2

TOPICS = ["ATP:0000041", "ATP:0000061", "ATP:0000062", "ATP:0000068", "ATP:0000069", "ATP:0000070",
          "ATP:0000082", "ATP:0000083", "ATP:0000084", "ATP:0000096", "ATP:0000152"]
NAMES = {"ATP:0000041": "gene expression (WT)", "ATP:0000061": "catalytic activity",
         "ATP:0000062": "transporter function", "ATP:0000068": "genetic interaction",
         "ATP:0000069": "physical interaction", "ATP:0000070": "regulatory interaction",
         "ATP:0000082": "RNAi phenotype", "ATP:0000083": "allele phenotype",
         "ATP:0000084": "overexpression phenotype", "ATP:0000096": "antibody", "ATP:0000152": "disease model"}

DATASET_SQL = """
SELECT DISTINCT ON (d.data_type) d.id, d.data_type
FROM dataset d JOIN mod m ON m.id = d.mod_id
WHERE m.abbreviation = 'WB' AND d.dataset_type = 'document' AND d.data_type = ANY(%s)
ORDER BY d.data_type, d.version DESC
"""


def load_answers(out_dir: str, model: str) -> Dict[str, Dict[str, bool]]:
    answers: Dict[str, Dict[str, bool]] = {}
    results_dir = os.path.join(out_dir, "results", model)
    for name in os.listdir(results_dir):
        record = json.load(open(os.path.join(results_dir, name)))
        if record.get("status") != "ok":
            continue
        answer = record["answer"]
        answers[record["curie"]] = {t: bool(answer[t.replace(":", "_")]["positive"]) for t in TOPICS}
    return answers


def load_dataset_labels() -> Dict[str, Dict[str, str]]:
    conn = psycopg2.connect(host=os.environ["PSQL_HOST"], port=os.environ.get("PSQL_PORT", "5432"),
                            dbname=os.environ["PSQL_DATABASE"], user=os.environ["PSQL_USERNAME"],
                            password=os.environ["PSQL_PASSWORD"])
    conn.set_session(readonly=True)
    labels: Dict[str, Dict[str, str]] = {t: {} for t in TOPICS}
    with conn, conn.cursor() as cur:
        cur.execute(DATASET_SQL, (TOPICS,))
        for dataset_id, topic in cur.fetchall():
            cur.execute("SELECT reference_curie, classification_value FROM dataset_entry WHERE dataset_id = %s",
                        (dataset_id,))
            for curie, value in cur.fetchall():
                if value in ("positive", "negative"):
                    labels[topic][curie] = value
    return labels


def corrected(raw: float, sens: float, spec: float) -> Optional[float]:
    denominator = sens + spec - 1
    if denominator <= 0.05:
        return None
    return min(1.0, max(0.0, (raw + spec - 1) / denominator))


def rates(pairs: List[Tuple[str, bool]]) -> Tuple[float, float, int, int]:
    pos = [llm for label, llm in pairs if label == "positive"]
    neg = [llm for label, llm in pairs if label == "negative"]
    sens = sum(pos) / len(pos) if pos else float("nan")
    spec = (len(neg) - sum(neg)) / len(neg) if neg else float("nan")
    return sens, spec, len(pos), len(neg)


def percentile(values: List[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def bucket_rows(topic: str, random_curies: List[str], answers: Dict[str, Dict[str, bool]],
                bucket_of: Dict[str, str], sens: float, spec: float, shrink: float, min_specificity: float,
                floor: float) -> List[list]:
    """Per-NNC-bucket positive share of one topic (see the module docstring)."""
    calls = [answers[c][topic] for c in random_curies]
    overall = sum(calls) / len(calls)
    by_bucket: Dict[str, List[bool]] = {}
    for curie in random_curies:
        bucket = bucket_of.get(curie)
        if bucket:
            by_bucket.setdefault(bucket, []).append(answers[curie][topic])
    rows = []
    for bucket in sorted(by_bucket):
        bucket_calls = by_bucket[bucket]
        raw = sum(bucket_calls) / len(bucket_calls)
        shrunk = (sum(bucket_calls) + shrink * overall) / (len(bucket_calls) + shrink)
        fixed = corrected(shrunk, sens, spec)
        if fixed is None:
            share, method = shrunk, "raw (no usable error rates)"
        elif spec < min_specificity:
            share, method = (shrunk + fixed) / 2, f"midpoint (specificity {spec:.2f} < {min_specificity})"
        else:
            share, method = fixed, "corrected"
        share = min(1 - floor, max(floor, share))
        rows.append([topic, bucket, len(bucket_calls), sum(bucket_calls), f"{raw:.4f}", f"{shrunk:.4f}",
                     "" if fixed is None else f"{fixed:.4f}", f"{share:.4f}", method])
    return rows


def load_buckets(csv_path: str) -> Dict[str, Dict[str, str]]:
    bucket_of: Dict[str, Dict[str, str]] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["nnc_bucket"] and row["nnc_bucket"] != "no NNC":
                bucket_of.setdefault(row["topic"], {})[row["reference_curie"]] = row["nnc_bucket"]
    return bucket_of


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("--model", default="claude-opus-5-5", help="whose answers to use (results/<model>/)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=6482)
    parser.add_argument("--nnc_strata_csv", default=None,
                        help="per-paper NNC buckets; also write the per-bucket shares for "
                             "--rebalance bucket_and_positive_mix")
    parser.add_argument("--shrink", type=float, default=10.0)
    parser.add_argument("--min_specificity", type=float, default=0.9)
    parser.add_argument("--floor", type=float, default=0.005)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    answers = load_answers(args.out_dir, args.model)
    dataset_labels = load_dataset_labels()
    with open(os.path.join(args.out_dir, "random_sample.tsv")) as fh:
        random_curies = [row["curie"] for row in csv.DictReader(fh, delimiter="\t") if row["curie"] in answers]
    with open(os.path.join(args.out_dir, "calibration_sample.tsv")) as fh:
        calibration_curies = {row["curie"] for row in csv.DictReader(fh, delimiter="\t")}

    rows = []
    bucket_of = load_buckets(args.nnc_strata_csv) if args.nnc_strata_csv else {}
    per_bucket: List[list] = []
    for topic in TOPICS:
        labelled = [c for c in calibration_curies | set(random_curies)
                    if c in answers and c in dataset_labels[topic]]
        pairs = [(dataset_labels[topic][c], answers[c][topic]) for c in labelled]
        sens, spec, n_pos, n_neg = rates(pairs)
        calls = [answers[c][topic] for c in random_curies]
        raw = sum(calls) / len(calls) if calls else float("nan")
        point = corrected(raw, sens, spec) if n_pos and n_neg else None
        boot = []
        for _ in range(args.bootstrap):
            b_calls = [rng.choice(calls) for _ in calls]
            b_pairs = [rng.choice(pairs) for _ in pairs]
            b_sens, b_spec, b_pos, b_neg = rates(b_pairs)
            if b_pos and b_neg:
                value = corrected(sum(b_calls) / len(b_calls), b_sens, b_spec)
                if value is not None:
                    boot.append(value)
        low, high = (percentile(boot, 0.025), percentile(boot, 0.975)) if len(boot) > args.bootstrap // 2 else ("", "")
        train_pos = sum(1 for v in dataset_labels[topic].values() if v == "positive")
        train_share = train_pos / len(dataset_labels[topic]) if dataset_labels[topic] else float("nan")
        rows.append([topic, NAMES[topic], len(calls), f"{raw:.3f}", n_pos, n_neg, f"{sens:.2f}", f"{spec:.2f}",
                     "" if point is None else f"{point:.3f}", "" if low == "" else f"{low:.3f}",
                     "" if high == "" else f"{high:.3f}", f"{train_share:.3f}"])
        if topic in bucket_of and n_pos and n_neg:
            per_bucket += bucket_rows(topic, random_curies, answers, bucket_of[topic], sens, spec,
                                      args.shrink, args.min_specificity, args.floor)

    header = ["topic", "name", "papers_read", "raw_share", "labelled_pos", "labelled_neg", "sensitivity",
              "specificity", "corrected_share", "ci95_low", "ci95_high", "training_set_share"]
    with open(os.path.join(args.out_dir, f"estimate_{args.model}.tsv"), "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)
    print("\t".join(header))
    for row in rows:
        print("\t".join(str(x) for x in row))
    if per_bucket:
        bucket_header = ["topic", "bucket", "papers", "llm_positive", "raw_share", "shrunk_share",
                         "corrected_share", "positive_share", "method"]
        with open(os.path.join(args.out_dir, f"bucket_prevalence_{args.model}.tsv"), "w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(bucket_header)
            writer.writerows(per_bucket)
        print()
        print("\t".join(bucket_header))
        for row in per_bucket:
            print("\t".join(str(x) for x in row))


if __name__ == "__main__":
    main()
