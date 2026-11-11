#!/usr/bin/env python3
"""SCRUM-6482: compare the rebalancing modes of one WB topic, with the WB NNC classifier as a baseline.

Reads the stats JSON the trainer writes for each run (RUNS_DIR/<topic>_<mode>/WB_ATP_<topic>_stats.json) and prints
one row per mode: the plain holdout F1, the holdout weighted to the estimated corpus at the default cut-off and at
the tuned cut-off, and the precision-recall area (average precision, AP) on the estimated corpus.

NNC rows: the NNC classifier called positive when its bucket is HIGH (and, as looser cut-offs, MEDIUM or above, LOW
or above), scored on the same holdout from the per-bucket counts in the stats of the run named by --nnc_from
(every mode but bucket_subsample shares that holdout). NNC's estimated-corpus numbers follow directly from the
corpus bucket mix and the estimated share of positives per bucket (--bucket_prevalence_tsv), the same estimates
the trained models are weighted with, so they compare like for like - but they are not independent of those
estimates.

Usage:
    python scripts/wb_rebalance_compare.py RUNS_DIR ATP:0000061 --bucket_prevalence_tsv bucket_prevalence.tsv
        [--modes none bucket_mix ...] [--nnc_from none]
"""
import argparse
import json
import os
from typing import Dict, List, Optional

from agr_document_classifier.rebalance import load_bucket_prevalence

ORDER = ["HIGH", "MEDIUM", "LOW", "NEG"]
MODES = ["none", "bucket_mix", "bucket_and_positive_mix", "bucket_and_positive_mix_balanced", "bucket_subsample"]


def load_stats(runs_dir: str, topic: str, mode: str) -> Optional[dict]:
    short = topic.replace("ATP:", "")
    path = os.path.join(runs_dir, f"{short}_{mode}", f"WB_{topic.replace(':', '_')}_stats.json")
    if not os.path.exists(path):
        return None
    stats = json.load(open(path))
    return stats.get("stats", stats)


def prf(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def nnc_rows(stats: dict, corpus_mix: Dict[str, float], share: Dict[str, float]) -> List[dict]:
    """NNC at each bucket cut-off: plain holdout metrics from the holdout's per-bucket counts, and estimated-corpus
    metrics from the corpus mix and estimated positive share per bucket; plus NNC's AP on the estimated corpus
    (its four buckets as the ranking)."""
    by_bucket = stats["test_by_nnc_bucket"]
    holdout_pos = sum(m["positives"] for m in by_bucket.values())
    buckets = [b for b in ORDER if b in corpus_mix and b in share]
    present = sum(corpus_mix[b] for b in buckets)
    est_pos = {b: corpus_mix[b] / present * share[b] for b in buckets}
    est_neg = {b: corpus_mix[b] / present * (1 - share[b]) for b in buckets}
    total_est_pos = sum(est_pos.values())
    ap, previous_recall = 0.0, 0.0
    rows = []
    for k in range(1, len(buckets)):
        called = buckets[:k]
        tp = sum(by_bucket.get(b, {}).get("positives", 0) for b in called)
        fp = sum(by_bucket.get(b, {}).get("n", 0) - by_bucket.get(b, {}).get("positives", 0) for b in called)
        plain = prf(tp, fp, holdout_pos - tp)
        e_tp = sum(est_pos[b] for b in called)
        estimated = prf(e_tp, sum(est_neg[b] for b in called), total_est_pos - e_tp)
        ap += (estimated["recall"] - previous_recall) * estimated["precision"]
        previous_recall = estimated["recall"]
        rows.append({"label": f"NNC: {called[-1]} or above" if k > 1 else "NNC: HIGH",
                     "plain_f1": plain["f1"], "estimated": estimated})
    ap += (1.0 - previous_recall) * total_est_pos  # calling every paper positive: precision = prevalence
    for row in rows:
        row["ap"] = ap
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir")
    parser.add_argument("topic")
    parser.add_argument("--bucket_prevalence_tsv", required=True)
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--nnc_from", default="none", help="run whose holdout the NNC rows are scored on")
    args = parser.parse_args()
    share = load_bucket_prevalence(args.bucket_prevalence_tsv, args.topic)

    header = (f"{'method':36s} {'plain F1':>8s} | {'est. corpus @ default cut-off':^24s} | "
              f"{'cut-off':>7s} {'est. corpus @ tuned cut-off':^24s} | {'AP':>5s}")
    print(header)
    print(f"{'':36s} {'':>8s} | {'P':>7s} {'R':>7s} {'F1':>7s} | {'':>7s} {'P':>7s} {'R':>7s} {'F1':>7s} | {'':>5s}")
    corpus_mix, nnc_stats = None, None
    for mode in args.modes:
        stats = load_stats(args.runs_dir, args.topic, mode)
        if stats is None:
            print(f"{mode:36s} (not finished)")
            continue
        corpus_mix = corpus_mix or (stats.get("rebalance") or {}).get("corpus_mix")
        if mode == args.nnc_from:
            nnc_stats = stats
        est = stats.get("test_estimated_corpus_weighted", {})
        tuned = stats.get("tuned_threshold", {})
        t_est = tuned.get("test_estimated_corpus", {})
        note = " *" if mode == "bucket_subsample" else ""
        print(f"{mode + note:36s} {stats['test_f1']:8.3f} | {est.get('precision', 0):7.2f} {est.get('recall', 0):7.2f} "
              f"{est.get('f1', 0):7.3f} | {tuned.get('threshold', 0):7.3f} {t_est.get('precision', 0):7.2f} "
              f"{t_est.get('recall', 0):7.2f} {t_est.get('f1', 0):7.3f} | "
              f"{tuned.get('test_estimated_corpus_average_precision', 0):5.3f}")
    if nnc_stats and corpus_mix:
        for row in nnc_rows(nnc_stats, corpus_mix, share):
            e = row["estimated"]
            print(f"{row['label']:36s} {row['plain_f1']:8.3f} | {e['precision']:7.2f} {e['recall']:7.2f} "
                  f"{e['f1']:7.3f} | {'-':>7s} {'':>7s} {'':>7s} {'':>7s} | {row['ap']:5.3f}")
    print("\n* bucket_subsample has its own (smaller) holdout; its numbers are not on the same papers.")


if __name__ == "__main__":
    main()
