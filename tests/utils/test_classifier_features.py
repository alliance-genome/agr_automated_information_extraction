"""Tests for the shared classifier feature builder (SCRUM-6132).

Problem #2: mean-pooled BioWordVec dilutes decisive/rare tokens. These tests pin
the behaviour of the optional, stateless feature blocks (max pooling + hashing
BoW) and guarantee that the mean-only path is byte-for-byte unchanged.
"""

import numpy as np
import scipy.sparse as sp
from gensim.models import KeyedVectors

from utils.embedding import (
    BOW_CONFIG,
    build_document_features,
    build_feature_matrix,
    get_document_embedding,
)


def _toy_model():
    """A tiny in-memory KeyedVectors stand-in for BioWordVec."""
    kv = KeyedVectors(vector_size=4)
    kv.add_vectors(
        ["gene", "expression", "control"],
        np.array([
            [1.0, 0.0, 2.0, -1.0],
            [0.0, 3.0, 1.0, 0.5],
            [-2.0, 1.0, 0.0, 4.0],
        ], dtype=np.float32),
    )
    return kv


# ---------------------------------------------------------------------------
# Regression: mean-only path must equal the current implementation exactly
# ---------------------------------------------------------------------------


def test_mean_only_matches_get_document_embedding():
    kv = _toy_model()
    text = "gene expression control"
    feats = build_document_features(kv, text)  # both flags default off
    expected = get_document_embedding(kv, text)
    assert isinstance(feats, np.ndarray)
    assert np.allclose(feats, expected)
    assert feats.shape == (4,)


# ---------------------------------------------------------------------------
# Max pooling
# ---------------------------------------------------------------------------


def test_max_pooling_appends_elementwise_max():
    kv = _toy_model()
    text = "gene expression control"
    feats = build_document_features(kv, text, use_max_pooling=True)
    assert feats.shape == (8,)
    # First block is the mean, unchanged.
    assert np.allclose(feats[:4], get_document_embedding(kv, text))
    # Second block is the element-wise max over the valid word vectors.
    stack = kv[["gene", "expression", "control"]]
    assert np.allclose(feats[4:], stack.max(axis=0))


# ---------------------------------------------------------------------------
# BoW (hashing)
# ---------------------------------------------------------------------------


def test_bow_is_sparse_with_expected_width_and_deterministic():
    kv = _toy_model()
    text = "gene expression control"
    feats = build_document_features(kv, text, use_bow=True)
    assert sp.issparse(feats)
    assert feats.shape == (1, 4 + BOW_CONFIG["n_features"])
    # Stateless / deterministic: same input -> identical output, no fitting.
    again = build_document_features(kv, text, use_bow=True)
    assert (feats != again).nnz == 0


def test_bow_captures_oov_token_that_embedding_drops():
    kv = _toy_model()
    base = build_document_features(kv, "gene", use_bow=True)
    with_oov = build_document_features(kv, "gene fbal0001234", use_bow=True)
    dim = 4
    # The embedding (mean) block is identical: the OOV token contributes nothing.
    assert np.allclose(base[:, :dim].toarray(), with_oov[:, :dim].toarray())
    # But the BoW block gains the OOV token -> strictly more non-zeros.
    assert with_oov[:, dim:].nnz > base[:, dim:].nnz


def test_bow_combines_with_max_pooling_widths():
    kv = _toy_model()
    feats = build_document_features(kv, "gene expression", use_max_pooling=True, use_bow=True)
    assert sp.issparse(feats)
    assert feats.shape == (1, 8 + BOW_CONFIG["n_features"])


# ---------------------------------------------------------------------------
# All-OOV document: validity gate relies on the mean block being zeros
# ---------------------------------------------------------------------------


def test_all_oov_document_has_zero_mean_block():
    kv = _toy_model()
    feats = build_document_features(kv, "zzz qqq", use_max_pooling=True)
    assert np.allclose(feats, np.zeros(8))


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------


def test_build_feature_matrix_dense_and_sparse():
    kv = _toy_model()
    dense = build_feature_matrix(kv, ["gene expression", "control gene"])
    assert isinstance(dense, np.ndarray)
    assert dense.shape == (2, 4)

    sparse = build_feature_matrix(kv, ["gene expression", "control gene"], use_bow=True)
    assert sp.issparse(sparse)
    assert sparse.shape == (2, 4 + BOW_CONFIG["n_features"])
