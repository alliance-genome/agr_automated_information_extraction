#!/usr/bin/env python3
"""SCRUM-6482: bootstrap confidence intervals for the rebalancing comparison of one WB topic.

Scores the saved models of every mode (RUNS_DIR/<topic>_<mode>/WB_ATP_<topic>_classifier.joblib) and the WB NNC
classifier on two test sets, then resamples each test set with replacement (the same draws for every method, so
the differences between methods are paired) to give a 95% interval for each score and for each difference:

- the curator holdout: rebuilt by running the trainer with the arguments after ``--`` up to the point where it
  splits the data (same papers, same order, same random_state as the runs), so every mode but bucket_subsample
  is scored on the same papers it was tested on. Scores: F1 weighted to the estimated corpus at the tuned cut-off
  from each run's stats, the plain F1 at the model's default cut-off and the AP weighted to the estimated corpus.
  The per-paper weights are those of the full holdout, kept fixed in the resamples.
- the LLM-read random sample (papers in the topic's WB document dataset left out, as in
  wb_score_random_sample.py): F1 at the default and the tuned cut-off and the AP, against the LLM's labels.

The bootstrap shows how far the scores could move by chance with test sets of this size; it does not change the
holdout split (that needs retraining on other splits).

Usage (needs the trainer's environment plus read access to the ABC database, as wb_score_random_sample.py):
    python scripts/wb_rebalance_bootstrap.py RUNS_DIR ATP:0000061 LLM_DIR --nnc_strata_csv wb_nnc_papers.csv
        --bucket_prevalence_tsv bucket_prevalence.tsv [--draws 2000] -- <trainer arguments of the runs>
"""
import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import utils.thread_limits  # noqa: F401  (import first: pins native threads to 1)
import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import agr_document_classifier.agr_document_classifier_trainer as trainer
from agr_document_classifier.rebalance import load_bucket_prevalence, prevalence_weights
from scripts.wb_score_random_sample import dataset_curies, llm_labels
from utils.abc_embeddings import get_profile_by_name, ABC_EMBEDDING_PROFILE

MODES = ["none", "bucket_mix", "bucket_and_positive_mix", "bucket_and_positive_mix_balanced"]
REFERENCES = ["bucket_mix", "bucket_and_positive_mix_balanced"]
RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NEG": 0}


class _Captured(BaseException):
    pass


def capture_training_data(trainer_argv: List[str]) -> dict:
    """What the trainer passes to model fitting (X, y, buckets, strata, positive_share, fixed_model, ...), got by
    running it with ``trainer_argv`` and stopping it there."""
    captured = {}

    def stop(X, y, *args, **kwargs):
        captured.update(kwargs, X=X, y=y)
        raise _Captured()

    trainer._select_and_fit_model = stop
    sys.argv = ["agr_document_classifier_trainer.py"] + trainer_argv
    try:
        trainer.main()
    except _Captured:
        pass
    return captured


def capture_holdout(trainer_argv: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], object]:
    """X, y and NNC bucket of the holdout rows, exactly as the trainer splits them, and the trainer's strata."""
    data = capture_training_data(trainer_argv)
    X, y, buckets = data["X"], data["y"], data["buckets"]
    _, X_test, _, y_test, _, idx_test = train_test_split(
        X, y, np.arange(X.shape[0]), test_size=0.20, stratify=y, random_state=42)
    return X_test, y_test, [buckets[i] for i in idx_test], data["strata"]


def f1(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    tp = float(np.sum(w * (y == 1) * (pred == 1)))
    fp = float(np.sum(w * (y == 0) * (pred == 1)))
    fn = float(np.sum(w * (y == 1) * (pred == 0)))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def load_model(runs_dir: str, topic: str, mode: str):
    run = os.path.join(runs_dir, f"{topic.replace('ATP:', '')}_{mode}")
    stem = f"WB_{topic.replace(':', '_')}"
    stats = json.load(open(os.path.join(run, f"{stem}_stats.json")))
    stats = stats.get("stats", stats)
    return joblib.load(os.path.join(run, f"{stem}_classifier.joblib")), stats["tuned_threshold"]["threshold"]


def method_outputs(models: Dict[str, tuple], X: np.ndarray, nnc: np.ndarray) -> Dict[str, dict]:
    """Per method: positive-class scores, default-cut-off calls and tuned-cut-off calls (None for NNC)."""
    out = {}
    for mode, (model, threshold) in models.items():
        scores = trainer._decision_scores(model, X)
        out[mode] = {"scores": scores, "default": model.predict(X).astype(int),
                     "tuned": (scores >= threshold).astype(int)}
    for cut, label in ((3, "NNC: HIGH"), (2, "NNC: MEDIUM+"), (1, "NNC: LOW+")):
        out[label] = {"scores": nnc.astype(float), "default": (nnc >= cut).astype(int), "tuned": None}
    return out


def bootstrap(y: np.ndarray, w: np.ndarray, outputs: Dict[str, dict], metrics: Dict[str, tuple], draws: int,
              seed: int = 0) -> Dict[str, Dict[str, np.ndarray]]:
    """metric name -> method -> [point estimate, draw 1, ..., draw N]. ``metrics``: name -> (kind, calls key)."""
    rng = np.random.default_rng(seed)
    samples = [np.arange(len(y))] + [rng.integers(0, len(y), len(y)) for _ in range(draws)]
    result = {name: {} for name in metrics}
    for name, (kind, key) in metrics.items():
        for method, out in outputs.items():
            if kind == "f1" and out[key] is None:
                continue
            values = []
            for rows in samples:
                if kind == "f1":
                    values.append(f1(y[rows], out[key][rows], w[rows]))
                elif y[rows].sum() == 0:
                    values.append(np.nan)
                else:
                    values.append(average_precision_score(y[rows], out["scores"][rows], sample_weight=w[rows]))
            result[name][method] = np.array(values)
    return result


def report(title: str, result: Dict[str, Dict[str, np.ndarray]], references: List[str] = REFERENCES) -> None:
    print(f"\n=== {title}")
    for name, by_method in result.items():
        print(f"\n{name}: estimate [95% interval]; difference from each reference "
              f"[95% interval] (share of draws where the method is better)")
        refs = [r for r in references if r in by_method]
        print(f"  {'method':34s} {'estimate':>22s} | " + " | ".join(f"{'vs ' + r:>40s}" for r in refs))
        for method, values in by_method.items():
            low, high = np.nanpercentile(values[1:], [2.5, 97.5])
            cells = []
            for ref in refs:
                if ref == method:
                    cells.append(f"{'-':>40s}")
                    continue
                diff = values - by_method[ref]
                d_low, d_high = np.nanpercentile(diff[1:], [2.5, 97.5])
                better = 100 * np.nanmean(diff[1:] > 0)
                cells.append(f"{diff[0]:+7.3f} [{d_low:+.3f}, {d_high:+.3f}] ({better:3.0f}%)".rjust(40))
            print(f"  {method:34s} {values[0]:7.3f} [{low:.3f}, {high:.3f}] | " + " | ".join(cells))


def main() -> None:
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir")
    parser.add_argument("topic")
    parser.add_argument("llm_dir")
    parser.add_argument("--nnc_strata_csv", required=True)
    parser.add_argument("--bucket_prevalence_tsv", required=True)
    parser.add_argument("--llm_model", default="claude-sonnet-5-5")
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--embedding_cache_dir", default=None)
    parser.add_argument("--embedding_profile", default=ABC_EMBEDDING_PROFILE)
    args = parser.parse_args(argv[:split])
    trainer_argv = argv[split + 1:]

    models = {mode: load_model(args.runs_dir, args.topic, mode) for mode in args.modes}
    share = load_bucket_prevalence(args.bucket_prevalence_tsv, args.topic)
    bucket_of = {}
    with open(args.nnc_strata_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["topic"] == args.topic and row["nnc_bucket"] not in ("", "no NNC"):
                bucket_of[row["reference_curie"]] = row["nnc_bucket"]

    # random sample first: it needs no trainer run
    labels = llm_labels(args.llm_dir, args.llm_model, args.topic)
    training = dataset_curies(args.topic)
    held_out = sorted(c for c in labels if c not in training)
    profile = get_profile_by_name(args.embedding_profile)
    kept: List[str] = []
    X_r, _ = trainer._build_abc_embedding_features({"negative": held_out}, "WB", use_bow=True,
                                                   profile_name=profile.name, version=profile.version,
                                                   kept_curies=kept, cache_dir=args.embedding_cache_dir, workers=12)
    y_r = np.array([labels[c] for c in kept])
    nnc_r = np.array([RANK.get(bucket_of.get(c, ""), 0) for c in kept])
    random_result = bootstrap(y_r, np.ones(len(y_r)), method_outputs(models, X_r, nnc_r),
                              {"F1 at default cut-off": ("f1", "default"),
                               "F1 at tuned cut-off": ("f1", "tuned"), "AP": ("ap", None)}, args.draws)

    X_h, y_h, buckets_h, strata = capture_holdout(trainer_argv)
    w_h = np.asarray(prevalence_weights(buckets_h, y_h, strata, share))
    nnc_h = np.array([RANK.get(b or "", 0) for b in buckets_h])
    holdout_outputs = method_outputs(models, X_h, nnc_h)
    holdout_result = bootstrap(y_h, w_h, holdout_outputs,
                               {"F1 est. corpus at tuned cut-off": ("f1", "tuned"),
                                "AP est. corpus": ("ap", None)}, args.draws)
    plain = bootstrap(y_h, np.ones(len(y_h)), holdout_outputs, {"plain F1 at default cut-off": ("f1", "default")},
                      args.draws)

    print(f"{args.topic}: holdout {len(y_h)} papers ({int(y_h.sum())} positive); random sample {len(y_r)} papers "
          f"({int(y_r.sum())} LLM positive); {args.draws} bootstrap draws")
    report("curator holdout", {**plain, **holdout_result})
    report("random sample (LLM labels)", random_result)


if __name__ == "__main__":
    main()
