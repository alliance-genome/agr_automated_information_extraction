"""SCRUM-6482: weighting a training set to the estimated corpus, per NNC bucket and class."""
import numpy as np
import pytest

from agr_document_classifier.rebalance import NO_BUCKET, Strata, prevalence_weights

STRATA = Strata(bucket_of={}, corpus_mix={"HIGH": 0.2, "NEG": 0.8})
# training set: HIGH 30+/10-, NEG 20+/40-, two unscored papers
BUCKETS = ["HIGH"] * 40 + ["NEG"] * 60 + [NO_BUCKET] * 2
LABELS = [1] * 30 + [0] * 10 + [1] * 20 + [0] * 40 + [1, 0]


def _cell_share(weights, bucket, label, buckets=BUCKETS, labels=LABELS):
    scored = [i for i, b in enumerate(buckets) if b != NO_BUCKET]
    total = sum(weights[i] for i in scored)
    return sum(weights[i] for i in scored if buckets[i] == bucket and labels[i] == label) / total


def test_cells_get_their_estimated_corpus_share():
    weights = prevalence_weights(BUCKETS, LABELS, STRATA, {"HIGH": 0.5, "NEG": 0.05})
    assert _cell_share(weights, "HIGH", 1) == pytest.approx(0.2 * 0.5)
    assert _cell_share(weights, "HIGH", 0) == pytest.approx(0.2 * 0.5)
    assert _cell_share(weights, "NEG", 1) == pytest.approx(0.8 * 0.05)
    assert _cell_share(weights, "NEG", 0) == pytest.approx(0.8 * 0.95)
    assert weights.mean() == pytest.approx(1.0)


def test_unscored_rows_keep_a_neutral_weight():
    weights = prevalence_weights(BUCKETS, LABELS, STRATA, {"HIGH": 0.5, "NEG": 0.05})
    raw_unscored = weights[-1] / weights.mean()
    assert weights[-1] == weights[-2] == pytest.approx(raw_unscored)


def test_bucket_without_estimate_keeps_its_own_positive_share():
    weights = prevalence_weights(BUCKETS, LABELS, STRATA, {"NEG": 0.05})
    high_pos, high_neg = _cell_share(weights, "HIGH", 1), _cell_share(weights, "HIGH", 0)
    assert high_pos + high_neg == pytest.approx(0.2)
    assert high_pos / (high_pos + high_neg) == pytest.approx(30 / 40)


def test_single_class_bucket_is_weighted_as_a_whole():
    buckets = ["HIGH"] * 10 + ["NEG"] * 10
    labels = [1] * 10 + [0] * 10
    weights = prevalence_weights(buckets, labels, STRATA, {"HIGH": 0.5, "NEG": 0.05})
    assert np.isfinite(weights).all()
    assert _cell_share(weights, "HIGH", 1, buckets, labels) == pytest.approx(0.2)


def test_balanced_classes_carry_equal_weight_and_keep_relative_sizes():
    from agr_document_classifier.rebalance import balance_classes
    weights = prevalence_weights(BUCKETS, LABELS, STRATA, {"HIGH": 0.5, "NEG": 0.05})
    balanced = balance_classes(weights, LABELS)
    labels = np.array(LABELS)
    assert balanced[labels == 1].sum() == pytest.approx(balanced[labels == 0].sum())
    assert balanced.mean() == pytest.approx(1.0)
    # within a class the ratios are unchanged (HIGH positive vs NEG positive)
    assert balanced[0] / balanced[40] == pytest.approx(weights[0] / weights[40])


def test_best_threshold_maximises_weighted_f1():
    from agr_document_classifier.agr_document_classifier_trainer import _best_threshold
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9])
    threshold, f1 = _best_threshold(y, scores, None)
    assert 0.7 < threshold <= 0.8 and f1 == pytest.approx(1.0)
    # weighting the 0.6/0.7 negatives away makes a lower cut-off just as good; the higher one is kept on ties
    threshold_w, _ = _best_threshold(y, scores, np.array([1, 1, 0, 0, 1, 1]))
    assert threshold_w <= 0.8


def test_tuned_threshold_stats_runs_on_a_small_model():
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from agr_document_classifier.agr_document_classifier_trainer import _tuned_threshold_stats
    rng = np.random.default_rng(0)
    y = np.array([1] * 50 + [0] * 50)
    X = sparse.csr_matrix(rng.normal(size=(100, 5)) + y[:, None])
    buckets = (["HIGH"] * 30 + ["NEG"] * 20) + (["HIGH"] * 10 + ["NEG"] * 40)
    idx_tv, idx_te = np.arange(0, 100, 1)[::2], np.arange(0, 100, 1)[1::2]
    model = LogisticRegression().fit(X[idx_tv], y[idx_tv])
    stats = _tuned_threshold_stats(model, X[idx_tv], y[idx_tv], X[idx_te], y[idx_te], idx_tv, idx_te, None,
                                   buckets, STRATA, {"HIGH": 0.5, "NEG": 0.05},
                                   StratifiedKFold(n_splits=3, shuffle=True, random_state=0))
    assert 0 <= stats["threshold"] <= 1
    assert set(stats["test_estimated_corpus"]) == {"precision", "recall", "f1"}
    assert 0 <= stats["test_estimated_corpus_average_precision"] <= 1


def test_fixed_model_skips_the_search_and_reports_cv_and_threshold():
    from scipy import sparse
    from agr_document_classifier.agr_document_classifier_trainer import _select_and_fit_model
    rng = np.random.default_rng(1)
    y = np.array([1] * 60 + [0] * 60)
    X = sparse.csr_matrix(rng.normal(size=(120, 6)) + y[:, None])
    buckets = ["HIGH"] * 40 + ["NEG"] * 20 + ["HIGH"] * 15 + ["NEG"] * 45
    weights = prevalence_weights(buckets, y, STRATA, {"HIGH": 0.5, "NEG": 0.05})
    model, stats = _select_and_fit_model(
        X, y, False, None, None, sample_weight=weights, buckets=buckets, strata=STRATA,
        positive_share={"HIGH": 0.5, "NEG": 0.05},
        fixed_model=("LGBMClassifier", {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 5}))
    assert stats["model_name"] == "LGBMClassifier"
    assert stats["best_params"]["n_estimators"] == 20
    assert stats["all_models_test_scores"] == {}
    assert 0 <= stats["average_f1"] <= 1
    assert "threshold" in stats["tuned_threshold"]
