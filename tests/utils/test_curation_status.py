"""SCRUM-5764: writing the "won't curate" curation status and the "outside of
scope" curation tag to the ABC.

A ZFIN reference the abstract classifier confidently calls a molecular-probe
paper is dropped from the acquisition workflow with two writes besides its topic
tag: a "won't manually index" (ATP:0000343) workflow tag and a "won't curate"
(ATP:0000299) curation_status row, both carrying the "outside of scope"
(ATP:0000209) curation tag. POST /curation_status/ and POST /workflow_tag/ both
accept that curation_tag field.

Like every other ABC writer here, these go through urllib.request.urlopen, so a
single reference's HTTP failure must never crash the surrounding batch.
"""

import io
import json
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from utils import abc_utils


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
# send_curation_status_to_abc
# ---------------------------------------------------------------------------


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_curation_status_posts_status_and_curation_tag(mock_urlopen, _tok):
    mock_urlopen.return_value = _FakeResponse(201)

    assert abc_utils.send_curation_status_to_abc(
        reference_curie="AGRKB:101000000000001",
        mod_abbreviation="ZFIN",
        topic="ATP:0000002",
        curation_status="ATP:0000299",
        curation_tag="ATP:0000209",
    ) is True

    request = mock_urlopen.call_args[0][0]
    assert request.full_url.endswith("/curation_status/")
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["reference_curie"] == "AGRKB:101000000000001"
    assert payload["mod_abbreviation"] == "ZFIN"
    assert payload["topic"] == "ATP:0000002"
    assert payload["curation_status"] == "ATP:0000299"
    assert payload["curation_tag"] == "ATP:0000209"


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_curation_status_omits_curation_tag_when_not_given(mock_urlopen, _tok):
    mock_urlopen.return_value = _FakeResponse(201)

    abc_utils.send_curation_status_to_abc(
        reference_curie="AGRKB:101000000000001",
        mod_abbreviation="ZFIN",
        topic="ATP:0000002",
        curation_status="ATP:0000299",
    )

    payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    assert "curation_tag" not in payload


@patch("utils.abc_utils.time.sleep")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_curation_status_4xx_does_not_raise_and_does_not_retry(
        mock_urlopen, _tok, mock_sleep):
    mock_urlopen.side_effect = _http_error(422, "already exists")

    assert abc_utils.send_curation_status_to_abc(
        "AGRKB:1", "ZFIN", "ATP:0000002", "ATP:0000299", "ATP:0000209") is False
    assert mock_urlopen.call_count == 1


@patch("utils.abc_utils.time.sleep")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_curation_status_5xx_retries_then_returns_false(
        mock_urlopen, _tok, mock_sleep):
    mock_urlopen.side_effect = _http_error(500)

    assert abc_utils.send_curation_status_to_abc(
        "AGRKB:1", "ZFIN", "ATP:0000002", "ATP:0000299", "ATP:0000209") is False
    assert mock_urlopen.call_count == 3


@patch("utils.abc_utils.time.sleep")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_curation_status_url_error_retries_then_returns_false(
        mock_urlopen, _tok, mock_sleep):
    mock_urlopen.side_effect = URLError("connection refused")

    assert abc_utils.send_curation_status_to_abc(
        "AGRKB:1", "ZFIN", "ATP:0000002", "ATP:0000299", "ATP:0000209") is False
    assert mock_urlopen.call_count == 3


@patch("utils.abc_utils.time.sleep")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_curation_status_5xx_then_201_succeeds(mock_urlopen, _tok, mock_sleep):
    mock_urlopen.side_effect = [_http_error(500), _FakeResponse(201)]

    assert abc_utils.send_curation_status_to_abc(
        "AGRKB:1", "ZFIN", "ATP:0000002", "ATP:0000299", "ATP:0000209") is True
    assert mock_urlopen.call_count == 2


# ---------------------------------------------------------------------------
# transition_workflow_tag — an automated transition, never a raw tag write (SCRUM-6845)
# ---------------------------------------------------------------------------

MANUAL_INDEXING = "ATP:0000273"


@patch("utils.abc_utils.get_current_workflow_status", return_value=None)
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_workflow_tag_is_an_automated_transition_with_curation_tag(mock_urlopen, _tok, _current):
    mock_urlopen.return_value = _FakeResponse(200)

    assert abc_utils.transition_workflow_tag(
        reference_curie="AGRKB:101000000000001",
        mod_abbreviation="ZFIN",
        workflow_tag_atp_id="ATP:0000343",
        workflow_process_atp_id=MANUAL_INDEXING,
        curation_tag="ATP:0000209",
    ) is True

    request = mock_urlopen.call_args[0][0]
    assert request.full_url.endswith("/workflow_tag/transition_to_workflow_status")
    payload = json.loads(request.data.decode("utf-8"))
    assert payload == {"curie_or_reference_id": "AGRKB:101000000000001", "mod_abbreviation": "ZFIN",
                       "new_workflow_tag_atp_id": "ATP:0000343", "transition_type": "automated",
                       "curation_tag": "ATP:0000209"}


@patch("utils.abc_utils.get_current_workflow_status", return_value=None)
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_workflow_tag_payload_has_no_curation_tag_without_one(mock_urlopen, _tok, _current):
    mock_urlopen.return_value = _FakeResponse(200)

    abc_utils.transition_workflow_tag("AGRKB:101000000000001", "FB", "ATP:0000359", MANUAL_INDEXING)

    payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    assert "curation_tag" not in payload


# ---------------------------------------------------------------------------
# Re-run safety: a reference already in the target state is not sent again, so a
# second pass does not burn retries or report a failure that did not happen.
# ---------------------------------------------------------------------------


@patch("utils.abc_utils.get_current_workflow_status", return_value="ATP:0000343")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_workflow_tag_already_in_state_is_a_no_op(mock_urlopen, _tok, _current):
    assert abc_utils.transition_workflow_tag("AGRKB:1", "ZFIN", "ATP:0000343", MANUAL_INDEXING,
                                             curation_tag="ATP:0000209") is True
    assert mock_urlopen.call_count == 0


@patch("utils.abc_utils.get_current_workflow_status", return_value=None)
@patch("utils.abc_utils.time.sleep")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_workflow_tag_4xx_is_not_retried(mock_urlopen, _tok, mock_sleep, _current):
    mock_urlopen.side_effect = _http_error(422, "Transition from X to Y is NOT in the transition table")

    assert abc_utils.transition_workflow_tag("AGRKB:1", "ZFIN", "ATP:0000343", MANUAL_INDEXING) is False
    assert mock_urlopen.call_count == 1


@patch("utils.abc_utils.get_current_workflow_status", return_value=None)
@patch("utils.abc_utils.time.sleep")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_workflow_tag_5xx_still_retries(mock_urlopen, _tok, mock_sleep, _current):
    mock_urlopen.side_effect = _http_error(502)

    assert abc_utils.transition_workflow_tag("AGRKB:1", "ZFIN", "ATP:0000343", MANUAL_INDEXING) is False
    assert mock_urlopen.call_count == 3


@patch("utils.abc_utils.time.sleep")
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
def test_curation_status_duplicate_warns_rather_than_errors(
        mock_urlopen, _tok, mock_sleep, caplog):
    """A curation_status row already exists for this reference/topic. We cannot
    tell from the response whether it is ours or a curator's, so we neither
    overwrite it nor report it as an error — it is a warning, and the caller
    leaves the reference alone.
    """
    mock_urlopen.side_effect = _http_error(
        422, 'Error creating curation_status: (psycopg2.errors.UniqueViolation) duplicate key '
             'value violates unique constraint "curation_status_unique"')

    with caplog.at_level("WARNING", logger="utils.abc_utils"):
        assert abc_utils.send_curation_status_to_abc(
            "AGRKB:1", "ZFIN", "ATP:0000002", "ATP:0000299", "ATP:0000209") is False

    records = [r for r in caplog.records if "curation status" in r.message.lower()]
    assert records, "expected a log line about the existing curation status"
    assert all(r.levelname == "WARNING" for r in records)
    assert "already" in records[0].message.lower()
