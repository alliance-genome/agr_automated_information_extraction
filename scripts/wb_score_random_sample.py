#!/usr/bin/env python3
"""SCRUM-6482: score trained WB topic classifiers, and the WB NNC classifier, on the LLM-read random sample.

The random sample (wb_llm_prevalence_sample.py) is drawn from the papers the ABC classifies, so scoring on it shows
how a model does on the papers it will really see, unlike the curator-labelled holdout. Labels are the LLM's
answers (wb_llm_prevalence_read.py), which are not perfect: for each topic the LLM's sensitivity and specificity
against curator labels (wb_llm_prevalence_estimate.py) say how far off they are. Papers in the topic's WB document
dataset (the training data) are left out, and so are papers the LLM could not read.

For each model (RUNS_DIR/<topic>_<mode>/WB_ATP_<topic>_classifier.joblib): precision, recall and F1 at the model's
default cut-off and at the tuned cut-off from its stats, and the average precision (AP). NNC: its HIGH, MEDIUM-or-
above and LOW-or-above cut-offs, and its AP with the four buckets as the ranking.

Needs the ABC API credentials (embeddings; --embedding_cache_dir reuses the trainer's cache) and read access to the
ABC database (PSQL_HOST, PSQL_PORT, PSQL_DATABASE, PSQL_USERNAME, PSQL_PASSWORD) for the dataset.

Usage:
    python scripts/wb_score_random_sample.py RUNS_DIR ATP:0000061 LLM_DIR --nnc_strata_csv wb_nnc_papers.csv
        [--llm_model claude-sonnet-5-5] [--modes none bucket_mix ...] [--embedding_cache_dir DIR]
"""
import argparse
import csv
import json
import os
from typing import Dict, List

import utils.thread_limits  # noqa: F401  (import first: pins native threads to 1)
import joblib
import numpy as np
import psycopg2
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from agr_document_classifier.agr_document_classifier_trainer import _build_abc_embedding_features, _decision_scores
from utils.abc_embeddings import get_profile_by_name, ABC_EMBEDDING_PROFILE

MODES = ["none", "bucket_mix", "bucket_and_positive_mix", "bucket_and_positive_mix_balanced", "bucket_subsample"]
DATASET_SQL = """
SELECT de.reference_curie FROM dataset d JOIN mod m ON m.id = d.mod_id JOIN dataset_entry de ON de.dataset_id = d.id
WHERE m.abbreviation = 'WB' AND d.dataset_type = 'document' AND d.data_type = %s
"""


def dataset_curies(topic: str) -> set:
    conn = psycopg2.connect(host=os.environ["PSQL_HOST"], port=os.environ.get("PSQL_PORT", "5432"),
                            dbname=os.environ["PSQL_DATABASE"], user=os.environ["PSQL_USERNAME"],
                            password=os.environ["PSQL_PASSWORD"])
    conn.set_session(readonly=True)
    with conn, conn.cursor() as cur:
        cur.execute(DATASET_SQL, (topic,))
        return {row[0] for row in cur.fetchall()}


def llm_labels(llm_dir: str, llm_model: str, topic: str) -> Dict[str, int]:
    labels = {}
    with open(os.path.join(llm_dir, "random_sample.tsv")) as fh:
        curies = [row["curie"] for row in csv.DictReader(fh, delimiter="\t")]
    for curie in curies:
        path = os.path.join(llm_dir, "results", llm_model, curie.replace(":", "_") + ".json")
        if os.path.exists(path):
            record = json.load(open(path))
            if record.get("status") == "ok":
                labels[curie] = int(bool(record["answer"][topic.replace(":", "_")]["positive"]))
    return labels


BANDS = [("NEG", 0.0, 0.5), ("LOW", 0.5, 0.667), ("MEDIUM", 0.667, 0.833), ("HIGH", 0.833, 1.01)]


def bands(y: np.ndarray, scores: np.ndarray) -> List[str]:
    """'papers (LLM positives, %)' per production confidence band, as get_confidence_level assigns them."""
    cells = []
    for _, low, high in BANDS:
        mask = (scores >= low) & (scores < high)
        share = 100 * y[mask].mean() if mask.any() else 0.0
        cells.append(f"{int(mask.sum())} ({int(y[mask].sum())}, {share:.0f}%)")
    return cells


def metrics(y: np.ndarray, pred: np.ndarray) -> str:
    return (f"{precision_score(y, pred, zero_division=0):6.2f} {recall_score(y, pred, zero_division=0):6.2f} "
            f"{f1_score(y, pred, zero_division=0):6.3f} {int(pred.sum()):5d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir")
    parser.add_argument("topic")
    parser.add_argument("llm_dir")
    parser.add_argument("--nnc_strata_csv", required=True)
    parser.add_argument("--llm_model", default="claude-sonnet-5-5")
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--embedding_cache_dir", default=None)
    parser.add_argument("--embedding_profile", default=ABC_EMBEDDING_PROFILE)
    args = parser.parse_args()

    labels = llm_labels(args.llm_dir, args.llm_model, args.topic)
    training = dataset_curies(args.topic)
    held_out = sorted(c for c in labels if c not in training)
    profile = get_profile_by_name(args.embedding_profile)
    kept: List[str] = []
    X, _ = _build_abc_embedding_features({"negative": held_out}, "WB", use_bow=True, profile_name=profile.name,
                                         version=profile.version, kept_curies=kept,
                                         cache_dir=args.embedding_cache_dir, workers=12)
    y = np.array([labels[c] for c in kept])
    print(f"{args.topic}: {len(labels)} random papers read by the LLM, {len(labels) - len(held_out)} in the "
          f"training dataset left out, {len(kept)} with embeddings; LLM positives {int(y.sum())} "
          f"({100 * y.mean():.1f}%)\n")
    print(f"{'method':36s} | {'default cut-off: P R F1 called':^30s} | {'tuned cut-off: P R F1 called':^30s} | AP")
    band_rows = []
    short = args.topic.replace("ATP:", "")
    file_stem = f"WB_{args.topic.replace(':', '_')}"
    for mode in args.modes:
        model_path = os.path.join(args.runs_dir, f"{short}_{mode}", f"{file_stem}_classifier.joblib")
        stats_path = os.path.join(args.runs_dir, f"{short}_{mode}", f"{file_stem}_stats.json")
        if not os.path.exists(model_path):
            print(f"{mode:36s} | (not finished)")
            continue
        model = joblib.load(model_path)
        stats = json.load(open(stats_path))
        stats = stats.get("stats", stats)
        scores = _decision_scores(model, X)
        default = model.predict(X)
        threshold = (stats.get("tuned_threshold") or {}).get("threshold")
        tuned = metrics(y, (scores >= threshold).astype(int)) if threshold is not None else "-"
        print(f"{mode:36s} | {metrics(y, default)} | {tuned} | {average_precision_score(y, scores):.3f}",
              flush=True)
        band_rows.append((mode, bands(y, scores)))

    bucket_of = {}
    with open(args.nnc_strata_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["topic"] == args.topic and row["nnc_bucket"] not in ("", "no NNC"):
                bucket_of[row["reference_curie"]] = row["nnc_bucket"]
    rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NEG": 0}
    nnc_scores = np.array([rank.get(bucket_of.get(c, ""), 0) for c in kept])
    for cut, label in ((3, "NNC: HIGH"), (2, "NNC: MEDIUM or above"), (1, "NNC: LOW or above")):
        print(f"{label:36s} | {metrics(y, (nnc_scores >= cut).astype(int))} | {'':30s} | "
              f"{average_precision_score(y, nnc_scores):.3f}")
    print(f"\n{sum(1 for c in kept if c not in bucket_of)} papers have no NNC score (counted as NNC negative).")
    print("\nPapers in each production confidence band (score 0.5/0.667/0.833), with the LLM positives among them:")
    print(f"{'method':36s} | " + " | ".join(f"{name:>16s}" for name, _, _ in BANDS))
    for mode, cells in band_rows:
        print(f"{mode:36s} | " + " | ".join(f"{cell:>16s}" for cell in cells))


if __name__ == "__main__":
    main()
