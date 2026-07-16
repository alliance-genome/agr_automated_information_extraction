from io import BytesIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils import abc_embeddings


def _parquet_bytes(rows):
    """Build a minimal ABC embedding parquet from ``rows`` of
    ``(embedding_list, is_document_level, content)``."""
    table = pa.table({
        "embedding": pa.array([r[0] for r in rows], type=pa.list_(pa.float32())),
        "is_document_level": pa.array([r[1] for r in rows], type=pa.bool_()),
        "content": pa.array([r[2] for r in rows], type=pa.string()),
    })
    buffer = BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def test_marker_round_trip_without_bow():
    parsed = abc_embeddings.parse_embedding_marker(abc_embeddings.format_embedding_marker())
    assert parsed is not None
    assert parsed["profile_name"] == abc_embeddings.ABC_EMBEDDING_PROFILE
    assert parsed["version"] == abc_embeddings.ABC_EMBEDDING_VERSION
    assert parsed["pooling"] == abc_embeddings.ABC_EMBEDDING_POOLING
    assert parsed["bow"] is False


def test_marker_round_trip_with_bow():
    parsed = abc_embeddings.parse_embedding_marker(
        abc_embeddings.format_embedding_marker(use_bow=True))
    assert parsed is not None
    assert parsed["bow"] is True


def test_marker_absent_returns_none():
    # Legacy (BioWordVec) models have no marker: description is None or unrelated.
    assert abc_embeddings.parse_embedding_marker(None) is None
    assert abc_embeddings.parse_embedding_marker("") is None
    assert abc_embeddings.parse_embedding_marker("some free-form description") is None


def test_pool_is_l2_normalized_chunk_mean_and_excludes_document_level():
    # Two paragraph rows + one document-level row (must be excluded from both the
    # pooled vector and the BoW text).
    parquet = _parquet_bytes([
        ([1.0, 0.0], False, "alpha beta"),
        ([0.0, 1.0], False, "gamma"),
        ([9.0, 9.0], True, "DOCLEVEL"),
    ])
    result = abc_embeddings.paragraph_pool_and_text(parquet)
    assert result is not None
    pooled, text = result
    # L2([1,0])=[1,0]; L2([0,1])=[0,1]; mean=[.5,.5]; L2(mean)=[0.7071,0.7071].
    np.testing.assert_allclose(pooled, np.array([0.70710678, 0.70710678], dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(pooled), 1.0, rtol=1e-5)
    assert text == "alpha beta gamma"
    assert pooled.dtype == np.float32


def test_pool_none_when_only_document_level():
    parquet = _parquet_bytes([([5.0, 6.0], True, "x")])
    assert abc_embeddings.paragraph_pool_and_text(parquet) is None


def test_pool_skips_null_embeddings_but_keeps_text():
    parquet = _parquet_bytes([
        (None, False, "no-vector text"),
        ([2.0, 0.0], False, "has vector"),
    ])
    pooled, text = abc_embeddings.paragraph_pool_and_text(parquet)
    # Only the one real vector contributes: L2([2,0]) = [1,0].
    np.testing.assert_allclose(pooled, np.array([1.0, 0.0], dtype=np.float32), rtol=1e-5)
    assert text == "no-vector text has vector"
