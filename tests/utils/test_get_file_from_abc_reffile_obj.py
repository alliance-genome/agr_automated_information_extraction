import os
import tempfile
from unittest.mock import MagicMock, patch

import requests

from utils import abc_utils


@patch("utils.abc_utils.generate_headers", return_value={})
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.requests.get")
def test_returns_content_on_200(mock_get, _tok, _hdr):
    resp = MagicMock()
    resp.content = b"# real markdown"
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp

    out = abc_utils.get_file_from_abc_reffile_obj({"referencefile_id": 1})

    assert out == b"# real markdown"
    # A request timeout must be honoured so a hung backend cannot block a batch.
    assert mock_get.call_args.kwargs["timeout"] == 60


@patch("utils.abc_utils.generate_headers", return_value={})
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.requests.get")
def test_returns_none_on_http_error(mock_get, _tok, _hdr):
    # A 4xx/5xx error body must NOT be returned as document content.
    resp = MagicMock()
    resp.content = b'{"detail":"Not Found"}'
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
    mock_get.return_value = resp

    assert abc_utils.get_file_from_abc_reffile_obj({"referencefile_id": 1}) is None


@patch("utils.abc_utils.generate_headers", return_value={})
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.requests.get")
def test_returns_none_on_timeout(mock_get, _tok, _hdr):
    mock_get.side_effect = requests.exceptions.Timeout("timed out")

    assert abc_utils.get_file_from_abc_reffile_obj({"referencefile_id": 1}) is None


@patch("utils.abc_utils.generate_headers", return_value={})
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.requests.get")
def test_returns_none_on_connection_error(mock_get, _tok, _hdr):
    mock_get.side_effect = requests.exceptions.ConnectionError("boom")

    assert abc_utils.get_file_from_abc_reffile_obj({"referencefile_id": 1}) is None


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
@patch("utils.abc_utils.get_file_from_abc_reffile_obj", return_value=None)
def test_download_main_pdf_writes_nothing_when_download_fails(
    _mock_download, _req, mock_urlopen, _tok,
):
    """When the download returns None (e.g. an HTTP error), download_main_pdf
    must not crash and must not write a bogus .pdf file."""
    import json

    ref_files = [
        {
            "file_class": "main",
            "file_publication_status": "final",
            "file_extension": "pdf",
            "referencefile_id": 1,
            "referencefile_mods": [{"mod_abbreviation": "WB"}],
        },
    ]
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = json.dumps(ref_files).encode("utf-8")
    mock_urlopen.return_value = cm

    with tempfile.TemporaryDirectory() as tmp:
        abc_utils.download_main_pdf("AGRKB:1", "WB", "AGRKB_1", tmp)
        assert os.listdir(tmp) == []
