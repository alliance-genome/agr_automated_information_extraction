"""Regression tests for the SCRUM-5716 strict-REST API migration (SCRUM-6163).

The send_*_to_abc / create_workflow_tag helpers POST via urllib.request.urlopen,
which raises urllib.error.HTTPError / URLError — not
requests.exceptions.RequestException. A single reference's failure must never
crash the surrounding job batch, and the server's idempotent "already exists"
responses must be treated as benign no-ops on re-runs.

Note: send_classification_tag_to_abc / send_entity_tag_to_abc are covered
separately in test_abc_utils_note.py (the SCRUM-5716 / PR #127 work).
"""

import io
import json
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from utils import abc_utils


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for the urlopen() context manager response."""

    def __init__(self, code: int = 201):
        self._code = code

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def getcode(self) -> int:
        return self._code


def _http_error(code: int, body: str = "") -> HTTPError:
    return HTTPError(
        url="http://abc/test",
        code=code,
        msg=f"HTTP {code}",
        hdrs=None,
        fp=io.BytesIO(body.encode("utf-8")),
    )


# ---------------------------------------------------------------------------
# send_manual_indexing_to_abc
# ---------------------------------------------------------------------------


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_manual_indexing_201_returns_true(mock_urlopen, _tok):
    mock_urlopen.return_value = _FakeResponse(201)
    assert abc_utils.send_manual_indexing_to_abc(
        "AGRKB:1", "FB", "ATP:0000207", 0.9
    ) is True
    assert mock_urlopen.call_count == 1


@patch("utils.abc_utils.time.sleep")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_manual_indexing_422_already_exists_no_crash_returns_false(
    mock_urlopen, _tok, mock_sleep,
):
    """A duplicate-row 422 must not crash the batch, but it must NOT be treated
    as success either: returning True would mask other validation failures and
    incorrectly advance failed jobs to done (SCRUM-6062). So: no crash, no
    retry, return False."""
    mock_urlopen.side_effect = _http_error(
        422,
        json.dumps({
            "detail": "ManualIndexingTag already exists for "
                      "reference_curie=AGRKB:1, mod_abbreviation=FB, "
                      "curation_tag=ATP:0000207"
        }),
    )
    assert abc_utils.send_manual_indexing_to_abc(
        "AGRKB:1", "FB", "ATP:0000207", 0.9
    ) is False
    # No retry on a deterministic 4xx.
    assert mock_urlopen.call_count == 1
    mock_sleep.assert_not_called()


@patch("utils.abc_utils.time.sleep")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_manual_indexing_422_other_returns_false(mock_urlopen, _tok, mock_sleep):
    mock_urlopen.side_effect = _http_error(
        422, json.dumps({"detail": "Reference with curie AGRKB:1 does not exist"}),
    )
    assert abc_utils.send_manual_indexing_to_abc(
        "AGRKB:1", "FB", "ATP:0000207", 0.9
    ) is False
    # Non-duplicate client error: no retry.
    assert mock_urlopen.call_count == 1
    mock_sleep.assert_not_called()


@patch("utils.abc_utils.time.sleep")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_manual_indexing_4xx_does_not_raise(mock_urlopen, _tok, mock_sleep):
    """The original bug: a 4xx HTTPError used to propagate uncaught and crash
    the whole batch."""
    mock_urlopen.side_effect = _http_error(400, "bad payload")
    # Must not raise.
    assert abc_utils.send_manual_indexing_to_abc(
        "AGRKB:1", "FB", "ATP:0000207", 0.9
    ) is False


@patch("utils.abc_utils.time.sleep")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_manual_indexing_5xx_retries_then_returns_false(
    mock_urlopen, _tok, mock_sleep,
):
    mock_urlopen.side_effect = _http_error(503, "upstream unavailable")
    assert abc_utils.send_manual_indexing_to_abc(
        "AGRKB:1", "FB", "ATP:0000207", 0.9
    ) is False
    assert mock_urlopen.call_count == 3
    # backoff sleeps between attempts 1 and 2, and 2 and 3.
    assert mock_sleep.call_count == 2


@patch("utils.abc_utils.time.sleep")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_manual_indexing_url_error_retries_then_returns_false(
    mock_urlopen, _tok, mock_sleep,
):
    mock_urlopen.side_effect = URLError("connection refused")
    assert abc_utils.send_manual_indexing_to_abc(
        "AGRKB:1", "FB", "ATP:0000207", 0.9
    ) is False
    assert mock_urlopen.call_count == 3
    assert mock_sleep.call_count == 2


@patch("utils.abc_utils.time.sleep")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_manual_indexing_5xx_then_201_succeeds(mock_urlopen, _tok, mock_sleep):
    mock_urlopen.side_effect = [
        _http_error(503, "upstream unavailable"),
        _FakeResponse(201),
    ]
    assert abc_utils.send_manual_indexing_to_abc(
        "AGRKB:1", "FB", "ATP:0000207", 0.9
    ) is True
    assert mock_urlopen.call_count == 2
    assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# get_pmids_from_reference_curies
# ---------------------------------------------------------------------------


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_get_pmids_one_bad_curie_does_not_crash_loop(mock_urlopen, _tok):
    """A 404 (or any HTTPError) for a single reference must not abort the whole
    loop; that curie maps to None and the rest are still processed."""
    good_body = json.dumps({
        "cross_references": [{"curie": "PMID:12345"}]
    }).encode("utf-8")

    class _ReadResp:
        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return self._body

    mock_urlopen.side_effect = [
        _http_error(404, json.dumps({"detail": "not found"})),
        _ReadResp(good_body),
    ]
    result = abc_utils.get_pmids_from_reference_curies(["AGRKB:bad", "AGRKB:good"])
    assert result == {"AGRKB:bad": None, "AGRKB:good": "PMID:12345"}
