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
from urllib.error import HTTPError

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
