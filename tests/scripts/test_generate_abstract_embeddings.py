from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

from scripts import generate_abstract_embeddings as gen
from utils.abc_embeddings import ABSTRACT_PROFILE, paragraph_pool_and_text


def test_parquet_round_trips_through_the_production_reader(tmp_path):
    # The whole point of writing ABC-format parquets is that the real consumer
    # reads them unchanged, so assert against paragraph_pool_and_text itself.
    vector = [0.0, 3.0, 4.0]
    with patch.object(gen, "embed_texts", return_value=[vector]):
        written = gen.generate(
            [{"curie": "AGRKB:1", "label": "positive",
              "title": "Probe synthesis", "abstract": "We made a probe."}],
            str(tmp_path))

    assert written == 1
    parquet_bytes = (tmp_path / "AGRKB:1.parquet").read_bytes()
    pooled, text = paragraph_pool_and_text(parquet_bytes)
    # A single chunk pools to its own L2-normalized vector: [0,3,4]/5.
    np.testing.assert_allclose(pooled, np.array([0.0, 0.6, 0.8], dtype=np.float32), rtol=1e-5)
    # The BoW block hashes exactly this text.
    assert text == "Probe synthesis\n\nWe made a probe."


def test_row_is_a_chunk_not_document_level(tmp_path):
    # A document-level row is skipped by the reader, which would make the
    # parquet unusable (paragraph_pool_and_text returns None).
    with patch.object(gen, "embed_texts", return_value=[[1.0, 0.0]]):
        gen.generate([{"curie": "AGRKB:2", "label": "negative",
                       "title": "T", "abstract": "A"}], str(tmp_path))

    table = pq.read_table(tmp_path / "AGRKB:2.parquet")
    assert table.column("is_document_level").to_pylist() == [False]
    assert table.column("chunk_index").to_pylist() == [0]
    assert table.column("profile_name").to_pylist() == [ABSTRACT_PROFILE.name]


def test_content_uses_the_shared_chunk_text_convention(tmp_path):
    # The text embedded and the text written to `content` must be the same
    # string, or the BoW block would be hashed from something the vector never
    # saw. Assert the embedder was handed exactly what landed in the parquet.
    with patch.object(gen, "embed_texts", return_value=[[1.0, 0.0]]) as mock_embed:
        gen.generate([{"curie": "AGRKB:3", "label": "positive",
                       "title": "  Padded title  ", "abstract": "  Padded abstract.  "}],
                     str(tmp_path))

    embedded_texts = mock_embed.call_args[0][0]
    assert embedded_texts == ["Padded title\n\nPadded abstract."]
    table = pq.read_table(tmp_path / "AGRKB:3.parquet")
    assert table.column("content").to_pylist() == embedded_texts


def test_generate_is_a_noop_on_an_empty_record_list(tmp_path):
    # A fully-filtered input must not fire an embeddings request with no inputs,
    # which the API rejects.
    with patch.object(gen, "embed_texts") as mock_embed:
        assert gen.generate([], str(tmp_path)) == 0
    mock_embed.assert_not_called()
