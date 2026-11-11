#!/usr/bin/env python3
"""SCRUM-6482: compare the rebalancing modes of one WB topic with k-fold cross-validation on all curator papers.

The runs score each mode on one 20% holdout. Here every curator paper is scored once, by a model trained on the
other folds, so the comparison uses all the papers and does not hang on one split. The training data (papers,
features, NNC buckets, estimated positive share per bucket, the fixed model) comes from running the trainer with
the arguments after ``--`` up to model fitting; each mode then weights the papers as train_classifier does
(weights computed on all papers, the training folds' slice used to fit, as the runs do with train+val).
bucket_subsample is left out: it trains on a different set of papers.

Scores per mode, all on the out-of-fold scores: plain F1 at the default cut-off (0.5) and plain AP; F1 weighted to the
estimated corpus at the default cut-off and at a tuned cut-off; AP weighted to the estimated corpus. The tuned
cut-off of each fold is chosen (best estimated-corpus F1, as the trainer does) on the out-of-fold scores of the
other folds only, so the papers it is applied to do not choose it. The WB NNC classifier is scored on the same
papers. The whole is repeated for each --seeds split; the first split also gets bootstrap 95% intervals for the
scores and for the paired differences between modes (wb_rebalance_bootstrap.py).

Usage (needs the trainer's environment):
    python scripts/wb_rebalance_kfold.py ATP:0000061 [--folds 5] [--seeds 42 1 2] [--draws 2000]
        -- <trainer arguments of the runs, with --fixed_model, --nnc_strata_csv and --bucket_prevalence_tsv>
"""
import argparse
import os
import sys
from typing import Dict, List

import utils.thread_limits  # noqa: F401  (import first: pins native threads to 1)
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

import agr_document_classifier.agr_document_classifier_trainer as trainer
from agr_document_classifier.rebalance import balance_classes, corpus_weights, prevalence_weights
from scripts.wb_rebalance_bootstrap import RANK, bootstrap, capture_training_data, f1, report

MODES = ["none", "bucket_mix", "bucket_and_positive_mix", "bucket_and_positive_mix_balanced"]


def mode_weights(mode: str, buckets: List[str], y: np.ndarray, strata, positive_share: dict):
    """The training weights train_classifier gives each paper in this mode."""
    if mode == "none":
        return None
    if mode == "bucket_mix":
        return np.asarray(corpus_weights(buckets, strata))
    weights = prevalence_weights(buckets, y, strata, positive_share)
    if mode == "bucket_and_positive_mix_balanced":
        weights = balance_classes(weights, y)
    return np.asarray(weights)


def cross_fitted_calls(y: np.ndarray, oof: np.ndarray, buckets: List[str], strata, positive_share: dict,
                       folds: List[np.ndarray]) -> np.ndarray:
    """Calls at a tuned cut-off, each fold's cut-off chosen on the other folds' out-of-fold scores."""
    calls = np.zeros(len(y), dtype=int)
    for test_rows in folds:
        other = np.setdiff1d(np.arange(len(y)), test_rows)
        weights = prevalence_weights([buckets[i] for i in other], y[other], strata, positive_share)
        threshold, _ = trainer._best_threshold(y[other], oof[other], weights)
        calls[test_rows] = (oof[test_rows] >= threshold).astype(int)
    return calls


def main() -> None:
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topic")
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args(argv[:split])

    data = capture_training_data(argv[split + 1:])
    X, y, buckets, strata = data["X"], data["y"], data["buckets"], data["strata"]
    share = data["positive_share"]
    name, params = data["fixed_model"]
    model = clone(trainer.POSSIBLE_CLASSIFIERS[name]["model"]).set_params(**params)
    eval_weights = np.asarray(prevalence_weights(buckets, y, strata, share))
    nnc = np.array([RANK.get(b or "", 0) for b in buckets])
    n_jobs = min(args.folds, int(os.environ.get("SEARCH_MAX_JOBS", str(args.folds))))
    print(f"{args.topic}: {len(y)} curator papers ({int(y.sum())} positive), {args.folds}-fold, "
          f"{name} {params}\n")

    point_rows: Dict[str, List[str]] = {}
    first_outputs = None
    for seed in args.seeds:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        folds = [test_rows for _, test_rows in splitter.split(X, y)]
        outputs = {}
        for mode in args.modes:
            fit_weight = mode_weights(mode, buckets, y, strata, share)
            oof, _ = trainer._cross_validate(model, X, y, fit_weight, splitter, n_jobs)
            outputs[mode] = {"scores": oof, "default": (oof >= 0.5).astype(int),
                             "tuned": cross_fitted_calls(y, oof, buckets, strata, share, folds)}
            print(f"  split {seed}: {mode} done", flush=True)
        for cut, label in ((3, "NNC: HIGH"), (2, "NNC: MEDIUM+"), (1, "NNC: LOW+")):
            outputs[label] = {"scores": nnc.astype(float), "default": (nnc >= cut).astype(int), "tuned": None}
        for method, out in outputs.items():
            ones = np.ones(len(y))
            tuned = f"{f1(y, out['tuned'], eval_weights):7.3f}" if out["tuned"] is not None else f"{'-':>7s}"
            point_rows.setdefault(method, []).append(
                f"{f1(y, out['default'], ones):7.3f} {f1(y, out['default'], eval_weights):7.3f} {tuned} "
                f"{average_precision_score(y, out['scores'], sample_weight=eval_weights):7.3f} "
                f"{average_precision_score(y, out['scores']):7.3f}")
        first_outputs = first_outputs or outputs

    print("\nPer split: plain F1 @0.5 | est. corpus F1 @0.5 | est. corpus F1 @tuned | est. corpus AP | plain AP")
    print(f"  {'method':34s} " + " | ".join(f"{'split ' + str(s):^39s}" for s in args.seeds))
    for method, cells in point_rows.items():
        print(f"  {method:34s} " + " | ".join(cells))

    ones = np.ones(len(y))
    result = bootstrap(y, eval_weights, first_outputs,
                       {"F1 est. corpus at tuned cut-off": ("f1", "tuned"),
                        "AP est. corpus": ("ap", None)}, args.draws)
    plain = bootstrap(y, ones, first_outputs, {"plain F1 at default cut-off": ("f1", "default")}, args.draws)
    report(f"out-of-fold scores, split {args.seeds[0]}, {args.draws} bootstrap draws", {**plain, **result})


if __name__ == "__main__":
    main()
