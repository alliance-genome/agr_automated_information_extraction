from io import BytesIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils import abc_embeddings


def _parquet_bytes(rows):
    """Build a minimal ABC embedding parquet from ``rows`` of
    ``(embedding_list, is_document_level)``."""
    table = pa.table({
        "embedding": pa.array([r[0] for r in rows], type=pa.list_(pa.float32())),
        "is_document_level": pa.array([r[1] for r in rows], type=pa.bool_()),
    })
    buffer = BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def test_marker_round_trip():
    marker = abc_embeddings.format_embedding_marker()
    parsed = abc_embeddings.parse_embedding_marker(marker)
    assert parsed is not None
    assert parsed["profile_name"] == abc_embeddings.ABC_EMBEDDING_PROFILE
    assert parsed["version"] == abc_embeddings.ABC_EMBEDDING_VERSION
    assert parsed["model"] == abc_embeddings.ABC_EMBEDDING_MODEL
    assert parsed["dim"] == abc_embeddings.ABC_EMBEDDING_DIM
    assert parsed["pooling"] == abc_embeddings.ABC_EMBEDDING_POOLING


def test_marker_absent_returns_none():
    # Legacy (BioWordVec) models have no marker: description is None or unrelated.
    assert abc_embeddings.parse_embedding_marker(None) is None
    assert abc_embeddings.parse_embedding_marker("") is None
    assert abc_embeddings.parse_embedding_marker("some free-form description") is None


def test_paragraph_mean_averages_only_paragraph_rows():
    # Two paragraph rows + one document-level row; the doc-level row must be ignored.
    parquet = _parquet_bytes([
        ([0.0, 0.0], False),
        ([2.0, 4.0], False),
        ([100.0, 100.0], True),  # document-level: excluded
    ])
    mean = abc_embeddings.paragraph_mean_from_parquet(parquet)
    assert mean is not None
    np.testing.assert_allclose(mean, np.array([1.0, 2.0], dtype=np.float32))
    assert mean.dtype == np.float32


def test_paragraph_mean_none_when_only_document_level():
    parquet = _parquet_bytes([([5.0, 6.0], True)])
    assert abc_embeddings.paragraph_mean_from_parquet(parquet) is None


def test_paragraph_mean_skips_null_embeddings():
    parquet = _parquet_bytes([
        (None, False),
        ([2.0, 2.0], False),
    ])
    mean = abc_embeddings.paragraph_mean_from_parquet(parquet)
    np.testing.assert_allclose(mean, np.array([2.0, 2.0], dtype=np.float32))
