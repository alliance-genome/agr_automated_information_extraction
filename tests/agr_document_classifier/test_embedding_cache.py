"""SCRUM-6482: training embeddings are kept on disk and reused, and parallel fetching keeps the row order."""
from unittest.mock import patch

import numpy as np

from agr_document_classifier import agr_document_classifier_trainer as trainer
from utils.abc_embeddings import ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION

CURIES = {"positive": ["AGRKB:1", "AGRKB:2"], "negative": ["AGRKB:3", "AGRKB:4"]}


def _fake_embedding(curie, _mod, profile_name=None, version=None):
    if curie == "AGRKB:4":
        return None
    number = int(curie.split(":")[1])
    return np.full(4, number, dtype=np.float32), f"paper {number} text"


def _build(cache_dir=None, workers=1):
    kept = []
    with patch.object(trainer, "get_reference_embedding", side_effect=_fake_embedding) as fetch:
        X, y = trainer._build_abc_embedding_features(CURIES, "WB", use_bow=True, kept_curies=kept,
                                                     cache_dir=cache_dir, workers=workers)
    return X, y, kept, fetch


def test_second_run_reads_from_the_cache(tmp_path):
    X1, y1, kept1, fetch1 = _build(cache_dir=str(tmp_path))
    assert fetch1.call_count == 4
    X2, y2, kept2, fetch2 = _build(cache_dir=str(tmp_path))
    # only the reference with no embedding is asked for again
    assert [c.args[0] for c in fetch2.call_args_list] == ["AGRKB:4"]
    assert y1 == y2 == [1, 1, 0] and kept1 == kept2 == ["AGRKB:1", "AGRKB:2", "AGRKB:3"]
    assert (X1 != X2).nnz == 0


def test_cache_is_keyed_by_profile_and_version(tmp_path):
    _build(cache_dir=str(tmp_path))
    path = trainer._cache_path(str(tmp_path), "AGRKB:1", ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION)
    pooled, text = trainer._read_cached_features(path)
    assert pooled.tolist() == [1, 1, 1, 1] and text == "paper 1 text"
    assert f"{ABC_EMBEDDING_PROFILE}_v{ABC_EMBEDDING_VERSION}" in path


def test_unreadable_cache_file_is_fetched_again(tmp_path):
    path = trainer._cache_path(str(tmp_path), "AGRKB:1", ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION)
    tmp_path.joinpath(path).parent.mkdir(parents=True)
    open(path, "w").write("not an npz")
    _, _, _, fetch = _build(cache_dir=str(tmp_path))
    assert "AGRKB:1" in [c.args[0] for c in fetch.call_args_list]
    assert trainer._read_cached_features(path) is not None


def test_parallel_fetch_keeps_the_row_order():
    X1, y1, kept1, _ = _build(workers=1)
    X8, y8, kept8, _ = _build(workers=8)
    assert y1 == y8 and kept1 == kept8 and (X1 != X8).nnz == 0
