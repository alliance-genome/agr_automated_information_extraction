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
    LSH_CONFIG,
    build_document_features,
    build_feature_matrix,
    get_document_embedding,
    lsh_feature_width,
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
# LSH bag-of-concepts (hash embeddings directly, no clustering)
# ---------------------------------------------------------------------------


def _directional_model():
    """A model whose words point in controlled directions, so LSH bucket
    assignment is fully deterministic:

    - ``same`` is a positive multiple of ``base`` (identical direction) -> every
      random hyperplane gives it the same sign -> identical buckets.
    - ``opposite`` is ``-base`` -> every sign flips -> a disjoint set of buckets.
    """
    base = np.array([0.7, -1.3, 0.4, 2.1, -0.9, 0.2, 1.1, -0.5], dtype=np.float32)
    kv = KeyedVectors(vector_size=8)
    kv.add_vectors(["base", "same", "opposite"],
                   np.array([base, 2.0 * base, -base], dtype=np.float32))
    return kv


def test_lsh_is_sparse_with_expected_width_and_deterministic():
    kv = _toy_model()
    text = "gene expression control"
    feats = build_document_features(kv, text, use_lsh=True)
    assert sp.issparse(feats)
    assert feats.shape == (1, 4 + lsh_feature_width())
    # Stateless / deterministic: same input -> identical output, no fitting.
    again = build_document_features(kv, text, use_lsh=True)
    assert (feats != again).nnz == 0
    # Mean block is preserved as the (dense) front of the row.
    assert np.allclose(feats[0, :4].toarray().ravel(), get_document_embedding(kv, text))


def test_lsh_collides_codirectional_words_into_same_buckets():
    kv = _directional_model()
    dim = 8
    n_tables = LSH_CONFIG["n_tables"]
    base_only = build_document_features(kv, "base", use_lsh=True)[:, dim:]
    co_directional = build_document_features(kv, "base same", use_lsh=True)[:, dim:]
    opposite = build_document_features(kv, "base opposite", use_lsh=True)[:, dim:]

    # A single word occupies exactly one bucket per table.
    assert base_only.nnz == n_tables
    # "same" points the same way as "base" -> they share every bucket: the set of
    # occupied buckets is unchanged, only the counts double.
    assert co_directional.nnz == n_tables
    assert co_directional.sum() == 2 * n_tables
    assert (co_directional.indices == base_only.indices).all()
    # "opposite" flips every sign -> a disjoint set of buckets, so nnz doubles.
    assert opposite.nnz == 2 * n_tables


def test_lsh_combines_with_bow_and_max_pooling_widths():
    kv = _toy_model()
    # LSH together with BoW (the "both" experiment).
    both = build_document_features(kv, "gene expression", use_lsh=True, use_bow=True)
    assert sp.issparse(both)
    assert both.shape == (1, 4 + lsh_feature_width() + BOW_CONFIG["n_features"])
    # Block order is [mean | max | lsh | bow]: mean stays first for the gate.
    full = build_document_features(kv, "gene expression", use_max_pooling=True,
                                   use_lsh=True, use_bow=True)
    assert full.shape == (1, 8 + lsh_feature_width() + BOW_CONFIG["n_features"])
    assert np.allclose(full[0, :4].toarray().ravel(), get_document_embedding(kv, "gene expression"))


def test_lsh_all_oov_document_is_zero_block_and_keeps_mean_gate():
    kv = _toy_model()
    feats = build_document_features(kv, "zzz qqq", use_lsh=True)
    assert feats.shape == (1, 4 + lsh_feature_width())
    # No in-vocabulary words -> empty LSH block and a zero mean block (the gate).
    assert feats.nnz == 0


def test_build_feature_matrix_lsh_is_sparse():
    kv = _toy_model()
    matrix = build_feature_matrix(kv, ["gene expression", "control gene"], use_lsh=True)
    assert sp.issparse(matrix)
    assert matrix.shape == (2, 4 + lsh_feature_width())


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
