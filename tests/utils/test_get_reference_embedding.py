from unittest.mock import patch

import numpy as np

from utils import abc_utils
from utils.abc_embeddings import ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION, ABSTRACT_PROFILE


def _embedding_row(referencefile_id, profile, version, source_file_class):
    return {
        "file_class": "embedding",
        "file_extension": "parquet",
        "referencefile_id": referencefile_id,
        "profile_name": profile,
        "version": version,
        "source": {"file_class": source_file_class} if source_file_class else None,
    }


@patch("utils.abc_utils.paragraph_pool_and_text",
       return_value=(np.array([1.0, 2.0], dtype=np.float32), "doc text"))
@patch("utils.abc_utils.get_file_from_abc_reffile_obj", return_value=b"parquet-bytes")
@patch("utils.abc_utils._show_all_for_reference")
def test_selects_main_source_matching_profile(mock_show_all, mock_download, mock_pool):
    mock_show_all.return_value = [
        # supplement-source embedding of the right profile → must be ignored
        _embedding_row(1, ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION, "converted_merged_supplement"),
        # wrong profile, main source → must be ignored
        _embedding_row(2, "some_other_profile", ABC_EMBEDDING_VERSION, "converted_merged_main"),
        # right profile + main source → the one to use
        _embedding_row(3, ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION, "converted_merged_main"),
    ]
    pooled, text = abc_utils.get_reference_embedding("AGRKB:1", "FB")
    np.testing.assert_allclose(pooled, np.array([1.0, 2.0], dtype=np.float32))
    assert text == "doc text"
    # It downloaded the main-source row (id 3), not the supplement or wrong-profile ones.
    assert mock_download.call_args[0][0]["referencefile_id"] == 3


@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
@patch("utils.abc_utils._show_all_for_reference")
def test_none_when_no_main_source_embedding(mock_show_all, mock_download):
    mock_show_all.return_value = [
        _embedding_row(1, ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION, "converted_merged_supplement"),
        {"file_class": "main", "file_extension": "pdf", "referencefile_id": 9},
    ]
    assert abc_utils.get_reference_embedding("AGRKB:1", "FB") is None
    mock_download.assert_not_called()


@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
@patch("utils.abc_utils._show_all_for_reference")
def test_none_when_wrong_version(mock_show_all, mock_download):
    mock_show_all.return_value = [
        _embedding_row(1, ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION + 1, "converted_merged_main"),
    ]
    assert abc_utils.get_reference_embedding("AGRKB:1", "FB") is None
    mock_download.assert_not_called()


@patch("utils.abc_utils._show_all_for_reference", return_value=None)
def test_none_when_show_all_fails(_mock_show_all):
    assert abc_utils.get_reference_embedding("AGRKB:1", "FB") is None


@patch("utils.abc_utils.paragraph_pool_and_text")
@patch("utils.abc_utils.get_file_from_abc_reffile_obj", return_value=b"")
@patch("utils.abc_utils._show_all_for_reference")
def test_none_when_download_empty(mock_show_all, _mock_download, mock_pool):
    mock_show_all.return_value = [
        _embedding_row(3, ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION, "converted_merged_main"),
    ]
    assert abc_utils.get_reference_embedding("AGRKB:1", "FB") is None
    mock_pool.assert_not_called()


@patch("utils.abc_utils.paragraph_pool_and_text",
       return_value=(np.array([0.0, 1.0], dtype=np.float32), "Title\n\nAbstract."))
@patch("utils.abc_utils.get_file_from_abc_reffile_obj", return_value=b"parquet-bytes")
@patch("utils.abc_utils._show_all_for_reference")
def test_abstract_profile_accepts_row_with_no_source(mock_show_all, mock_download, _mock_pool):
    # An abstract embedding has no source referencefile, so requiring
    # converted_merged_main would reject it outright.
    mock_show_all.return_value = [
        _embedding_row(7, ABSTRACT_PROFILE.name, ABSTRACT_PROFILE.version, None),
    ]
    result = abc_utils.get_reference_embedding(
        "AGRKB:1", "ZFIN",
        profile_name=ABSTRACT_PROFILE.name, version=ABSTRACT_PROFILE.version)
    assert result is not None
    pooled, text = result
    np.testing.assert_allclose(pooled, np.array([0.0, 1.0], dtype=np.float32))
    assert text == "Title\n\nAbstract."
    assert mock_download.call_args[0][0]["referencefile_id"] == 7


@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
@patch("utils.abc_utils._show_all_for_reference")
def test_abstract_profile_ignores_fulltext_rows(mock_show_all, mock_download):
    # Both profiles are 1536-d, so picking the wrong one would NOT raise — it
    # would silently predict from the wrong features. The profile name is the
    # only thing preventing that.
    mock_show_all.return_value = [
        _embedding_row(1, ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION, "converted_merged_main"),
    ]
    assert abc_utils.get_reference_embedding(
        "AGRKB:1", "ZFIN",
        profile_name=ABSTRACT_PROFILE.name, version=ABSTRACT_PROFILE.version) is None
    mock_download.assert_not_called()


@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
@patch("utils.abc_utils._show_all_for_reference")
def test_fulltext_profile_still_rejects_sourceless_rows(mock_show_all, mock_download):
    # The fulltext constraint must not be loosened by making it per-profile.
    mock_show_all.return_value = [
        _embedding_row(1, ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION, None),
    ]
    assert abc_utils.get_reference_embedding("AGRKB:1", "FB") is None
    mock_download.assert_not_called()


@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
@patch("utils.abc_utils._show_all_for_reference")
def test_unknown_profile_requires_main_source(mock_show_all, mock_download):
    # An unregistered profile must not be treated as "no constraint" — that would
    # make a typo in a model's embedding_profile silently match anything.
    mock_show_all.return_value = [
        _embedding_row(1, "unregistered_profile", 1, None),
    ]
    assert abc_utils.get_reference_embedding(
        "AGRKB:1", "FB", profile_name="unregistered_profile", version=1) is None
    mock_download.assert_not_called()
