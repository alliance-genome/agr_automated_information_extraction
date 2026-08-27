"""The word embedding model must not be loaded unless a legacy classifier needs it.

BioWordVec is ~13 GB and takes minutes to load. Only a model without an
``embedding_profile`` pools word vectors; every model the trainer has produced
since SCRUM-5781 carries a profile and rebuilds its features from that instead,
so the common case must load nothing at all.
"""

import logging
from unittest.mock import patch

import numpy as np
import pytest

from agr_document_classifier import agr_document_classifier_classify as classify
from utils.embedding import LazyEmbeddingModel


class _Exploding:
    """Stands in for the loader where the code must never materialise the model."""

    def get(self):
        raise AssertionError("the embedding model was loaded on a path that cannot use it")


def test_constructing_the_loader_does_not_load():
    with patch("utils.embedding.load_embedding_model") as mock_load:
        LazyEmbeddingModel("/data/BioWordVec.vec.bin")
    mock_load.assert_not_called()


def test_get_loads_once_and_caches():
    with patch("utils.embedding.load_embedding_model", return_value="MODEL") as mock_load:
        loader = LazyEmbeddingModel("/data/BioWordVec.vec.bin")
        first = loader.get()
        second = loader.get()
    assert first == "MODEL"
    assert second is first
    # Loading per batch instead of per process would be worse than loading eagerly.
    assert mock_load.call_count == 1


def test_get_without_a_path_raises_a_named_error():
    loader = LazyEmbeddingModel(None)
    with pytest.raises(ValueError, match="embedding_model_path"):
        loader.get()


def test_a_missing_path_warns_but_does_not_raise(caplog):
    """A run covering only profile-carrying models never reads the file, so an
    absent path must not be fatal -- raising here would defeat the deferral."""
    with caplog.at_level(logging.WARNING, logger="utils.embedding"):
        LazyEmbeddingModel("/nonexistent/BioWordVec.vec.bin")
    assert "not found" in caplog.text


def test_a_failed_load_is_not_retried():
    """Several legacy topics in one run must not each re-attempt a multi-GB load."""
    with patch("utils.embedding.load_embedding_model",
               side_effect=OSError("boom")) as mock_load:
        loader = LazyEmbeddingModel(__file__)  # a path that exists, so no warning
        for _ in range(3):
            with pytest.raises(OSError, match="boom"):
                loader.get()
    assert mock_load.call_count == 1


@patch("agr_document_classifier.agr_document_classifier_classify.classify_documents_from_abc_embeddings")
def test_abc_embedding_path_never_loads_the_model(mock_classify):
    """A profile-carrying model classifies from its profile, so the loader stays untouched."""
    mock_classify.return_value = (["AGRKB:1"], np.array([1]), [0.9], [True])
    # test_mode=True keeps this off the network: it logs instead of sending TETs.
    classify.process_job_batch(
        [{"reference_curie": "AGRKB:1"}], "ZFIN", "ATP:0000370", 1, _Exploding(), object(),
        {"embedding_profile": "classifier_abstract_title_abstract_bow_only",
         "embedding_version": 1,
         "data_novelty": "ATP:0000335"},
        True, use_abc_embeddings=True, abc_use_bow=True)
    assert mock_classify.called
