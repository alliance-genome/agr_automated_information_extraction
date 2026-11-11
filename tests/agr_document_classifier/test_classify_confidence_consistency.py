"""Regression tests for SCRUM-6241.

``classify_documents`` used to derive the label from ``predict()`` and the
confidence from ``predict_proba()`` independently. For ``SVC(probability=True)``
those two mechanisms can disagree near the decision boundary (``predict()`` uses
the sign of the raw SVM margin, ``predict_proba()`` uses a separately fitted
Platt-scaling model), producing a *positive* classification with
``confidence_score < 0.5`` (documented scikit-learn behavior).

These tests pin the invariant: the label and the confidence score always agree.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC, LinearSVC

import pytest

from agr_document_classifier.agr_document_classifier_classify import (predict_labels_and_confidence,
                                                                      rescale_to_threshold)


def _overlapping_dataset():
    """A deliberately overlapping 2-class dataset. With ``random_state=0`` a
    fitted ``SVC(probability=True)`` produces several samples where the raw
    ``predict()`` label disagrees with ``predict_proba()`` around 0.5."""
    return make_classification(
        n_samples=200, n_features=8, n_informative=4, n_redundant=0,
        class_sep=0.3, flip_y=0.25, weights=[0.5, 0.5], random_state=0,
    )


def test_svc_label_and_confidence_always_agree():
    """Acceptance criteria: no positive with score < 0.5, no negative with
    score >= 0.5, for ``SVC(probability=True)``."""
    X, y = _overlapping_dataset()
    model = SVC(probability=True, random_state=42)
    model.fit(X, y)

    classifications, confidence_scores = predict_labels_and_confidence(model, X)

    assert len(classifications) == len(confidence_scores) == len(X)
    for label, score in zip(classifications, confidence_scores):
        if label > 0:
            assert score >= 0.5, f"positive classification with confidence {score} < 0.5"
        else:
            assert score < 0.5, f"negative classification with confidence {score} >= 0.5"


def test_helper_resolves_a_real_svc_disagreement():
    """Guard against a regression to ``predict()``-derived labels: on this fixed
    dataset the raw ``SVC.predict()`` disagrees with ``predict_proba()`` for at
    least one sample, and the helper must side with the confidence score."""
    X, y = _overlapping_dataset()
    model = SVC(probability=True, random_state=42)
    model.fit(X, y)

    raw_pred = model.predict(X)
    proba_positive = model.predict_proba(X)[:, 1]
    positive_label = model.classes_[1]
    disagreements = (raw_pred == positive_label) != (proba_positive >= 0.5)
    assert disagreements.any(), "test fixture no longer reproduces the SVC disagreement"

    classifications, confidence_scores = predict_labels_and_confidence(model, X)
    for idx in np.flatnonzero(disagreements):
        # The helper follows the confidence score, not the raw margin sign.
        expected = positive_label if proba_positive[idx] >= 0.5 else model.classes_[0]
        assert classifications[idx] == expected
        assert (classifications[idx] > 0) == (confidence_scores[idx] >= 0.5)


def test_linear_svc_decision_function_fallback_is_consistent():
    """LinearSVC has no ``predict_proba``; the helper falls back to
    ``decision_function`` -> sigmoid, which is inherently consistent."""
    X, y = _overlapping_dataset()
    model = LinearSVC(random_state=42, max_iter=5000)
    model.fit(X, y)

    classifications, confidence_scores = predict_labels_and_confidence(model, X)

    assert len(classifications) == len(confidence_scores) == len(X)
    for label, score in zip(classifications, confidence_scores):
        assert (label > 0) == (score >= 0.5)


def test_decision_threshold_rescales_the_score_so_the_threshold_is_0_5():
    """SCRUM-6482: ml_model.decision_threshold moves the cut-off and, by default, the
    confidence is rescaled so a positive always has at least 0.5."""
    X, y = _overlapping_dataset()
    model = SVC(probability=True, random_state=42).fit(X, y)
    _, raw_scores = predict_labels_and_confidence(model, X)
    classifications, scores = predict_labels_and_confidence(model, X, decision_threshold=0.3)

    assert scores == [rescale_to_threshold(s, 0.3) for s in raw_scores]
    for label, raw, score in zip(classifications, raw_scores, scores):
        assert (label > 0) == (raw >= 0.3) == (score >= 0.5)


def test_rescale_confidence_false_keeps_the_model_score():
    """ZFIN molecular probe: the label follows the threshold, the score stays the model's own."""
    X, y = _overlapping_dataset()
    model = SVC(probability=True, random_state=42).fit(X, y)
    _, raw_scores = predict_labels_and_confidence(model, X)
    classifications, scores = predict_labels_and_confidence(model, X, decision_threshold=0.8,
                                                            rescale_confidence=False)

    assert scores == raw_scores
    for label, score in zip(classifications, scores):
        assert (label > 0) == (score >= 0.8)


@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.87])
def test_rescale_to_threshold(threshold):
    assert rescale_to_threshold(threshold, threshold) == 0.5
    assert rescale_to_threshold(0.0, threshold) == 0.0
    assert rescale_to_threshold(1.0, threshold) == 1.0
    grid = np.linspace(0, 1, 101)
    rescaled = [rescale_to_threshold(s, threshold) for s in grid]
    assert all(a < b for a, b in zip(rescaled, rescaled[1:]))
    if threshold == 0.5:
        assert np.allclose(rescaled, grid)


def test_no_decision_threshold_means_0_5():
    X, y = _overlapping_dataset()
    model = LinearSVC(random_state=42, max_iter=5000).fit(X, y)
    default = predict_labels_and_confidence(model, X)
    explicit = predict_labels_and_confidence(model, X, decision_threshold=0.5)
    assert list(default[0]) == list(explicit[0])
    assert default[1] == explicit[1]
