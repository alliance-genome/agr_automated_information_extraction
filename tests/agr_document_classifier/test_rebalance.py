"""SCRUM-6482: rebalancing a WB training set to the corpus NNC-bucket mix."""
import csv
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from agr_document_classifier import agr_document_classifier_trainer as trainer
from agr_document_classifier.rebalance import (NO_BUCKET, Strata, bucket_counts, corpus_weights, load_strata,
                                               subsample)

TOPIC = "ATP:0000061"


def _write_papers(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reference_curie", "topic", "nnc_bucket", "classification", "year"])
        writer.writerows(rows)


def _corpus(tmp_path):
    """Corpus: 20 HIGH, 80 NEG (plus an old HIGH paper and another topic's paper)."""
    rows = [[f"AGRKB:H{i}", TOPIC, "HIGH", "", 2010] for i in range(20)]
    rows += [[f"AGRKB:N{i}", TOPIC, "NEG", "", 2010] for i in range(80)]
    rows += [["AGRKB:OLD", TOPIC, "HIGH", "", 1999], ["AGRKB:OTHER", "ATP:0000041", "LOW", "", 2010]]
    path = tmp_path / "papers.csv"
    _write_papers(path, rows)
    return str(path)


def _training():
    """Training: 16 HIGH (12+/4-), 8 NEG (2+/6-), 1 paper NNC never scored."""
    positive = [f"AGRKB:H{i}" for i in range(12)] + ["AGRKB:N0", "AGRKB:N1"] + ["AGRKB:UNSCORED"]
    negative = [f"AGRKB:H{i}" for i in range(12, 16)] + [f"AGRKB:N{i}" for i in range(2, 8)]
    return {"positive": positive, "negative": negative}


def test_load_strata(tmp_path):
    strata = load_strata(_corpus(tmp_path), TOPIC, year_from=2005)
    assert strata.corpus_mix == {"HIGH": 0.2, "NEG": 0.8}
    assert strata.bucket("AGRKB:H3") == "HIGH"
    assert strata.bucket("AGRKB:OTHER") == NO_BUCKET
    # without the date filter the old paper counts too
    assert load_strata(_corpus(tmp_path), TOPIC).corpus_mix["HIGH"] == pytest.approx(21 / 101)
    with pytest.raises(ValueError):
        load_strata(_corpus(tmp_path), "ATP:0000999")


def test_bucket_counts(tmp_path):
    strata = load_strata(_corpus(tmp_path), TOPIC, year_from=2005)
    assert bucket_counts(_training(), strata) == {
        "HIGH": {"positive": 12, "negative": 4},
        "NEG": {"positive": 2, "negative": 6},
        NO_BUCKET: {"positive": 1, "negative": 0}}


def test_subsample_matches_the_corpus_mix(tmp_path):
    strata = load_strata(_corpus(tmp_path), TOPIC, year_from=2005)
    sampled = subsample(_training(), strata, seed=1)
    counts = bucket_counts(sampled, strata)
    # NEG limits it: 8 NEG at an 80% share -> 10 papers, 2 of them HIGH; the unscored paper stays
    assert sum(counts["HIGH"].values()) == 2
    assert sum(counts["NEG"].values()) == 8
    assert counts[NO_BUCKET] == {"positive": 1, "negative": 0}
    # reproducible per seed, and only drops papers
    assert subsample(_training(), strata, seed=1) == sampled
    training = _training()
    for label in ("positive", "negative"):
        assert set(sampled[label]) <= set(training[label])


def test_corpus_weights():
    strata = Strata(bucket_of={}, corpus_mix={"HIGH": 0.2, "NEG": 0.8})
    buckets = ["HIGH"] * 16 + ["NEG"] * 8 + [NO_BUCKET]
    weights = corpus_weights(buckets, strata)
    assert weights.mean() == pytest.approx(1.0)
    # HIGH is 2/3 of the scored rows for a 20% share, NEG 1/3 for 80%: NEG weighs 8x HIGH
    assert weights[-2] / weights[0] == pytest.approx(8.0)
    # weighted, the scored rows have the corpus mix
    scored = weights[:-1]
    assert scored[:16].sum() / scored.sum() == pytest.approx(0.2)


def _toy_problem(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, 5)) + y[:, None]
    buckets = ["HIGH" if i % 4 == 0 else "NEG" for i in range(n)]
    return X, y, buckets


_SMALL_CLASSIFIERS = {
    "LogisticRegression": {"model": LogisticRegression(max_iter=200), "params": {"C": [0.1, 1.0]}},
    "KNeighborsClassifier": {"model": KNeighborsClassifier(), "params": {"n_neighbors": [3, 5]}},
}


@patch.dict(trainer.POSSIBLE_CLASSIFIERS, _SMALL_CLASSIFIERS, clear=True)
def test_select_and_fit_with_weights_and_bucket_metrics():
    X, y, buckets = _toy_problem()
    strata = Strata(bucket_of={}, corpus_mix={"HIGH": 0.1, "NEG": 0.9})
    weights = corpus_weights(buckets, strata)
    _model, stats = trainer._select_and_fit_model(X, y, False, "isolation_forest", 0.1,
                                                  sample_weight=weights, buckets=buckets, strata=strata)
    # KNN's fit takes no sample weights, so only LogisticRegression is tried
    assert stats["model_name"] == "LogisticRegression"
    assert set(stats["all_models_test_scores"]) == {"LogisticRegression"}
    assert stats["sample_weighted"] is True
    assert set(stats["test_by_nnc_bucket"]) == {"HIGH", "NEG"}
    assert sum(m["n"] for m in stats["test_by_nnc_bucket"].values()) == 40
    assert set(stats["test_corpus_weighted"]) == {"precision", "recall", "f1"}


@patch.dict(trainer.POSSIBLE_CLASSIFIERS, _SMALL_CLASSIFIERS, clear=True)
def test_select_and_fit_unweighted_keeps_every_model_and_the_split():
    X, y, buckets = _toy_problem()
    strata = Strata(bucket_of={}, corpus_mix={"HIGH": 0.1, "NEG": 0.9})
    _model, stats = trainer._select_and_fit_model(X, y, False, "isolation_forest", 0.1,
                                                  buckets=buckets, strata=strata)
    assert set(stats["all_models_test_scores"]) == {"LogisticRegression", "KNeighborsClassifier"}
    assert stats["sample_weighted"] is False
    # the holdout is the same 80/20 split as without buckets
    _train, test_idx = train_test_split(np.arange(len(y)), test_size=0.20, stratify=y, random_state=42)
    expected_high = sum(1 for i in test_idx if buckets[i] == "HIGH")
    assert stats["test_by_nnc_bucket"]["HIGH"]["n"] == expected_high


def test_outlier_removal_is_refused_with_strata():
    X, y, buckets = _toy_problem()
    strata = Strata(bucket_of={}, corpus_mix={"HIGH": 0.1, "NEG": 0.9})
    with pytest.raises(ValueError):
        trainer._select_and_fit_model(X, y, True, "isolation_forest", 0.1, buckets=buckets, strata=strata)
