"""Tests for the classifier model zoo (POSSIBLE_CLASSIFIERS).

LinearSVC and LGBMClassifier were added as sparse-friendly models that can run
on the high-dimensional BoW feature matrix (where rbf-SVC and MLP are skipped).
"""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV

from agr_document_classifier.models import POSSIBLE_CLASSIFIERS


def test_new_sparse_friendly_models_present():
    assert "LinearSVC" in POSSIBLE_CLASSIFIERS
    assert "LGBMClassifier" in POSSIBLE_CLASSIFIERS


@pytest.mark.parametrize("name", ["LinearSVC", "LGBMClassifier"])
def test_new_models_search_fits_on_sparse_input(name):
    """Every sampled hyperparameter combination must be valid and the model
    must fit/predict on a sparse (BoW-like) matrix."""
    rng = np.random.RandomState(0)
    X = sp.csr_matrix(rng.rand(40, 60))
    y = np.array([0, 1] * 20)
    info = POSSIBLE_CLASSIFIERS[name]
    search = RandomizedSearchCV(
        info["model"], info["params"], n_iter=5, cv=2,
        scoring=make_scorer(f1_score, zero_division=0),
        random_state=42, n_jobs=1,
    )
    search.fit(X, y)
    preds = search.predict(X)
    assert len(preds) == 40
