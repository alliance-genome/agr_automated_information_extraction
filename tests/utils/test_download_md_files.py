import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from utils import abc_utils


def _show_all_response(ref_files):
    payload = json.dumps(ref_files).encode("utf-8")
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = payload
    return cm


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
def test_downloads_only_main_md_for_matching_mod(
    mock_download, _req, mock_urlopen, _tok,
):
    ref_files = [
        # main MD for WB → should be downloaded
        {
            "file_extension": "md",
            "file_class": "converted_merged_main",
            "referencefile_id": 111,
            "referencefile_mods": [{"mod_abbreviation": "WB"}],
        },
        # supplement MD → must be skipped (include_supplements default=False)
        {
            "file_extension": "md",
            "file_class": "converted_merged_supplement",
            "referencefile_id": 222,
            "referencefile_mods": [{"mod_abbreviation": "WB"}],
        },
        # main MD but for a different MOD → must be skipped
        {
            "file_extension": "md",
            "file_class": "converted_merged_main",
            "referencefile_id": 333,
            "referencefile_mods": [{"mod_abbreviation": "MGI"}],
        },
        # TEI file → unused when MD exists for the same MOD
        {
            "file_extension": "tei",
            "file_class": "tei",
            "referencefile_id": 444,
            "referencefile_mods": [{"mod_abbreviation": "WB"}],
        },
    ]
    mock_urlopen.return_value = _show_all_response(ref_files)
    mock_download.return_value = b"# Downloaded markdown"

    with tempfile.TemporaryDirectory() as tmp:
        abc_utils.download_md_files_for_references(
            ["AGRKB:101000000000001"], tmp, "WB",
        )

        # Only one download call, and only for the matching main md.
        assert mock_download.call_count == 1
        called_ref = mock_download.call_args[0][0]
        assert called_ref["referencefile_id"] == 111

        out_path = os.path.join(tmp, "AGRKB_101000000000001.md")
        assert os.path.exists(out_path)
        with open(out_path, "rb") as fh:
            assert fh.read() == b"# Downloaded markdown"


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
def test_falls_back_to_tei_and_converts_when_no_main_md(
    mock_download, _req, mock_urlopen, _tok,
):
    """When no main MD exists but a TEI is present, the function should
    download the TEI bytes and convert them to MD via the shared library."""
    ref_files = [
        {
            "file_extension": "tei",
            "file_class": "tei",
            "referencefile_id": 444,
            "referencefile_mods": [{"mod_abbreviation": "WB"}],
        },
    ]
    mock_urlopen.return_value = _show_all_response(ref_files)
    tei_bytes = b"<?xml version='1.0'?><TEI xmlns='http://www.tei-c.org/ns/1.0'/>"
    mock_download.return_value = tei_bytes

    with patch("agr_abc_document_parsers.convert_xml_to_markdown",
               return_value="# Converted from TEI") as mock_convert:
        with tempfile.TemporaryDirectory() as tmp:
            abc_utils.download_md_files_for_references(
                ["AGRKB:101000000000001"], tmp, "WB",
            )

            mock_convert.assert_called_once_with(tei_bytes, "tei")
            out_path = os.path.join(tmp, "AGRKB_101000000000001.md")
            assert os.path.exists(out_path)
            with open(out_path, "r", encoding="utf-8") as fh:
                assert fh.read() == "# Converted from TEI"


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
def test_no_md_or_tei_writes_no_file(mock_download, _req, mock_urlopen, _tok):
    ref_files = [
        {
            "file_extension": "md",
            "file_class": "converted_merged_supplement",
            "referencefile_id": 999,
            "referencefile_mods": [{"mod_abbreviation": "WB"}],
        },
    ]
    mock_urlopen.return_value = _show_all_response(ref_files)

    with tempfile.TemporaryDirectory() as tmp:
        abc_utils.download_md_files_for_references(
            ["AGRKB:101000000000001"], tmp, "WB",
        )
        assert mock_download.call_count == 0
        assert os.listdir(tmp) == []


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
def test_only_first_main_md_is_taken(mock_download, _req, mock_urlopen, _tok):
    ref_files = [
        {
            "file_extension": "md",
            "file_class": "converted_merged_main",
            "referencefile_id": 1,
            "referencefile_mods": [{"mod_abbreviation": "WB"}],
        },
        {
            "file_extension": "md",
            "file_class": "converted_merged_main",
            "referencefile_id": 2,
            "referencefile_mods": [{"mod_abbreviation": "WB"}],
        },
    ]
    mock_urlopen.return_value = _show_all_response(ref_files)
    mock_download.return_value = b"# md"

    with tempfile.TemporaryDirectory() as tmp:
        abc_utils.download_md_files_for_references(["AGRKB:1"], tmp, "WB")
        assert mock_download.call_count == 1
        assert mock_download.call_args[0][0]["referencefile_id"] == 1


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
def test_supplements_downloaded_when_opted_in(
    mock_download, _req, mock_urlopen, _tok,
):
    """include_supplements=True should also download every
    converted_merged_supplement row for the same MOD, and supplements
    must NOT be converted from any other format."""
    ref_files = [
        {
            "file_extension": "md",
            "file_class": "converted_merged_main",
            "referencefile_id": 1,
            "referencefile_mods": [{"mod_abbreviation": "WB"}],
        },
        {
            "file_extension": "md",
            "file_class": "converted_merged_supplement",
            "referencefile_id": 11,
            "referencefile_mods": [{"mod_abbreviation": "WB"}],
        },
        {
            "file_extension": "md",
            "file_class": "converted_merged_supplement",
            "referencefile_id": 12,
            "referencefile_mods": [{"mod_abbreviation": "WB"}],
        },
        # Wrong MOD supplement → must be skipped
        {
            "file_extension": "md",
            "file_class": "converted_merged_supplement",
            "referencefile_id": 99,
            "referencefile_mods": [{"mod_abbreviation": "MGI"}],
        },
    ]
    mock_urlopen.return_value = _show_all_response(ref_files)

    def _fake_dl(ref):
        return ("# main" if ref["referencefile_id"] == 1
                else f"# supp {ref['referencefile_id']}").encode("utf-8")
    mock_download.side_effect = _fake_dl

    with tempfile.TemporaryDirectory() as tmp:
        abc_utils.download_md_files_for_references(
            ["AGRKB:101000000000001"], tmp, "WB", include_supplements=True,
        )

        files = sorted(os.listdir(tmp))
        assert files == [
            "AGRKB_101000000000001.md",
            "AGRKB_101000000000001.supp_1.md",
            "AGRKB_101000000000001.supp_2.md",
        ]
        with open(os.path.join(tmp, "AGRKB_101000000000001.supp_1.md"), "rb") as fh:
            assert fh.read() == b"# supp 11"
        with open(os.path.join(tmp, "AGRKB_101000000000001.supp_2.md"), "rb") as fh:
            assert fh.read() == b"# supp 12"


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
def test_supplements_not_downloaded_by_default(
    mock_download, _req, mock_urlopen, _tok,
):
    ref_files = [
        {
            "file_extension": "md",
            "file_class": "converted_merged_main",
            "referencefile_id": 1,
            "referencefile_mods": [{"mod_abbreviation": "WB"}],
        },
        {
            "file_extension": "md",
            "file_class": "converted_merged_supplement",
            "referencefile_id": 11,
            "referencefile_mods": [{"mod_abbreviation": "WB"}],
        },
    ]
    mock_urlopen.return_value = _show_all_response(ref_files)
    mock_download.return_value = b"# main"

    with tempfile.TemporaryDirectory() as tmp:
        abc_utils.download_md_files_for_references(
            ["AGRKB:1"], tmp, "WB",
        )
        assert os.listdir(tmp) == ["AGRKB_1.md"]
        # Only the main MD download was called; supplement was skipped.
        assert mock_download.call_count == 1
        assert mock_download.call_args[0][0]["referencefile_id"] == 1


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
def test_global_main_md_is_used_when_no_mod_scoped_md(
    mock_download, _req, mock_urlopen, _tok,
):
    """ABC stores converted_merged_main MDs globally (mod_id IS NULL,
    surfaced as mod_abbreviation: None). The downloader must select that
    row for any requested MOD instead of falling back to TEI conversion."""
    ref_files = [
        {
            "file_extension": "md",
            "file_class": "converted_merged_main",
            "referencefile_id": 500,
            "referencefile_mods": [{"mod_abbreviation": None}],
        },
        # A MOD-scoped TEI exists too; it must NOT be used because MD wins.
        {
            "file_extension": "tei",
            "file_class": "tei",
            "referencefile_id": 600,
            "referencefile_mods": [{"mod_abbreviation": "FB"}],
        },
    ]
    mock_urlopen.return_value = _show_all_response(ref_files)
    mock_download.return_value = b"# global md"

    with tempfile.TemporaryDirectory() as tmp:
        abc_utils.download_md_files_for_references(
            ["AGRKB:101000000000077"], tmp, "FB",
        )

        assert mock_download.call_count == 1
        assert mock_download.call_args[0][0]["referencefile_id"] == 500
        out_path = os.path.join(tmp, "AGRKB_101000000000077.md")
        assert os.path.exists(out_path)
        with open(out_path, "rb") as fh:
            assert fh.read() == b"# global md"


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
def test_main_md_with_empty_referencefile_mods_is_used(
    mock_download, _req, mock_urlopen, _tok,
):
    """Defensive: a converted_merged_main row whose referencefile_mods is
    an empty list is also a global file and must be selectable."""
    ref_files = [
        {
            "file_extension": "md",
            "file_class": "converted_merged_main",
            "referencefile_id": 700,
            "referencefile_mods": [],
        },
    ]
    mock_urlopen.return_value = _show_all_response(ref_files)
    mock_download.return_value = b"# empty-mods md"

    with tempfile.TemporaryDirectory() as tmp:
        abc_utils.download_md_files_for_references(["AGRKB:1"], tmp, "FB")
        assert mock_download.call_count == 1
        assert mock_download.call_args[0][0]["referencefile_id"] == 700


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
def test_main_md_scoped_to_other_mod_only_is_skipped(
    mock_download, _req, mock_urlopen, _tok,
):
    """A converted_merged_main row scoped exclusively to a different MOD
    must NOT be selected, even when the requested MOD has no MD. The
    downloader should fall through to the TEI fallback for the requested
    MOD."""
    ref_files = [
        {
            "file_extension": "md",
            "file_class": "converted_merged_main",
            "referencefile_id": 800,
            "referencefile_mods": [{"mod_abbreviation": "MGI"}],
        },
        {
            "file_extension": "tei",
            "file_class": "tei",
            "referencefile_id": 801,
            "referencefile_mods": [{"mod_abbreviation": "FB"}],
        },
    ]
    mock_urlopen.return_value = _show_all_response(ref_files)
    tei_bytes = b"<TEI/>"
    mock_download.return_value = tei_bytes

    with patch("agr_abc_document_parsers.convert_xml_to_markdown",
               return_value="# from tei") as mock_convert:
        with tempfile.TemporaryDirectory() as tmp:
            abc_utils.download_md_files_for_references(
                ["AGRKB:1"], tmp, "FB",
            )
            # The MGI-only MD must not be downloaded; the FB TEI must be.
            assert mock_download.call_count == 1
            assert mock_download.call_args[0][0]["referencefile_id"] == 801
            mock_convert.assert_called_once_with(tei_bytes, "tei")


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
def test_global_supplement_md_is_downloaded(
    mock_download, _req, mock_urlopen, _tok,
):
    """converted_merged_supplement rows are stored under the same global
    convention as main MDs; they must be downloaded when in scope (mod-
    scoped or global) for the requested MOD."""
    ref_files = [
        {
            "file_extension": "md",
            "file_class": "converted_merged_main",
            "referencefile_id": 900,
            "referencefile_mods": [{"mod_abbreviation": None}],
        },
        {
            "file_extension": "md",
            "file_class": "converted_merged_supplement",
            "referencefile_id": 901,
            "referencefile_mods": [{"mod_abbreviation": None}],
        },
        # Wrong-MOD supplement → must be skipped
        {
            "file_extension": "md",
            "file_class": "converted_merged_supplement",
            "referencefile_id": 902,
            "referencefile_mods": [{"mod_abbreviation": "MGI"}],
        },
    ]
    mock_urlopen.return_value = _show_all_response(ref_files)

    def _fake_dl(ref):
        return (f"# id-{ref['referencefile_id']}").encode("utf-8")
    mock_download.side_effect = _fake_dl

    with tempfile.TemporaryDirectory() as tmp:
        abc_utils.download_md_files_for_references(
            ["AGRKB:1"], tmp, "FB", include_supplements=True,
        )
        files = sorted(os.listdir(tmp))
        assert files == ["AGRKB_1.md", "AGRKB_1.supp_1.md"]
        with open(os.path.join(tmp, "AGRKB_1.supp_1.md"), "rb") as fh:
            assert fh.read() == b"# id-901"


def test_reffile_matches_mod_helper():
    """Direct coverage for the in-scope predicate."""
    matches = abc_utils._reffile_matches_mod

    # Mod-scoped to requested MOD
    assert matches({"referencefile_mods": [{"mod_abbreviation": "FB"}]}, "FB")
    # Global (mod_abbreviation: None) → in scope for any MOD
    assert matches({"referencefile_mods": [{"mod_abbreviation": None}]}, "FB")
    assert matches({"referencefile_mods": [{"mod_abbreviation": None}]}, "WB")
    # Empty referencefile_mods list → treated as global
    assert matches({"referencefile_mods": []}, "FB")
    # Missing referencefile_mods key → treated as global
    assert matches({}, "FB")
    # Mixed: one entry global, one entry different MOD → still in scope
    assert matches({"referencefile_mods": [{"mod_abbreviation": "MGI"},
                                           {"mod_abbreviation": None}]}, "FB")
    # Mixed: requested MOD present alongside another → in scope
    assert matches({"referencefile_mods": [{"mod_abbreviation": "MGI"},
                                           {"mod_abbreviation": "FB"}]}, "FB")
    # Scoped exclusively to a different MOD → out of scope
    assert not matches({"referencefile_mods": [{"mod_abbreviation": "MGI"}]}, "FB")
    # Multiple different MODs, none matching, none global → out of scope
    assert not matches({"referencefile_mods": [{"mod_abbreviation": "MGI"},
                                               {"mod_abbreviation": "WB"}]}, "FB")
