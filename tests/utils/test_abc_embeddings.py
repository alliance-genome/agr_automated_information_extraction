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


def test_recipe_fields():
    recipe = abc_embeddings.abc_embedding_recipe(use_bow=True)
    assert recipe["embedding_profile"] == abc_embeddings.ABC_EMBEDDING_PROFILE
    assert recipe["embedding_version"] == abc_embeddings.ABC_EMBEDDING_VERSION
    assert recipe["embedding_model"] == abc_embeddings.ABC_EMBEDDING_MODEL
    assert recipe["embedding_dim"] == abc_embeddings.ABC_EMBEDDING_DIM
    assert recipe["embedding_pooling"] == abc_embeddings.ABC_EMBEDDING_POOLING
    assert recipe["use_bow_features"] is True
    assert abc_embeddings.abc_embedding_recipe(use_bow=False)["use_bow_features"] is False


def test_is_abc_embedding_model():
    # ABC-embedding model: embedding_profile is set.
    assert abc_embeddings.is_abc_embedding_model({"embedding_profile": abc_embeddings.ABC_EMBEDDING_PROFILE})
    # Legacy / unavailable: profile null or absent.
    assert not abc_embeddings.is_abc_embedding_model({"embedding_profile": None})
    assert not abc_embeddings.is_abc_embedding_model({})
    assert not abc_embeddings.is_abc_embedding_model(None)


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
