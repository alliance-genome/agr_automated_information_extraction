import json
from urllib.error import HTTPError
from unittest.mock import patch

import pytest

from utils import abc_utils


def _read_payload_from_request(mock_request_cls):
    """Pull the JSON body out of the urlopen Request object the SUT built."""
    args, kwargs = mock_request_cls.call_args
    data = kwargs.get("data") or args[1]
    return json.loads(data.decode("utf-8"))


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_note_is_included_when_provided(mock_request, mock_urlopen, _tok):
    mock_urlopen.return_value.__enter__.return_value.getcode.return_value = 201

    abc_utils.send_classification_tag_to_abc(
        reference_curie="AGRKB:101000000000001",
        species="NCBITaxon:6239",
        topic="ATP:0000096",
        negated=False,
        data_novelty="ATP:0000335",
        confidence_score=None,
        confidence_level=None,
        tet_source_id=42,
        note="anti-PDR-1, raised antibody",
    )

    payload = _read_payload_from_request(mock_request)
    assert payload["note"] == "anti-PDR-1, raised antibody"


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_note_is_omitted_when_none(mock_request, mock_urlopen, _tok):
    mock_urlopen.return_value.__enter__.return_value.getcode.return_value = 201

    abc_utils.send_classification_tag_to_abc(
        reference_curie="AGRKB:101000000000001",
        species="NCBITaxon:6239",
        topic="ATP:0000096",
        negated=True,
        data_novelty="ATP:0000335",
        confidence_score=None,
        confidence_level=None,
        tet_source_id=42,
        note=None,
    )

    payload = _read_payload_from_request(mock_request)
    assert "note" not in payload


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_200_upsert_returns_true_without_error_log(mock_request, mock_urlopen, _tok, caplog):
    mock_urlopen.return_value.__enter__.return_value.getcode.return_value = 200

    result = abc_utils.send_classification_tag_to_abc(
        reference_curie="AGRKB:101000000000001",
        species="NCBITaxon:6239",
        topic="ATP:0000096",
        negated=False,
        data_novelty="ATP:0000335",
        confidence_score=None,
        confidence_level=None,
        tet_source_id=42,
    )

    assert result is True
    assert "Failed to create TET" not in caplog.text


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_409_duplicate_returns_true(mock_request, mock_urlopen, _tok):
    mock_urlopen.side_effect = HTTPError(
        url="http://x/topic_entity_tag/", code=409, msg="Conflict", hdrs=None, fp=None)

    result = abc_utils.send_classification_tag_to_abc(
        reference_curie="AGRKB:101000000000001",
        species="NCBITaxon:6239",
        topic="ATP:0000096",
        negated=False,
        data_novelty="ATP:0000335",
        confidence_score=None,
        confidence_level=None,
        tet_source_id=42,
    )

    assert result is True
    assert mock_urlopen.call_count == 1


@patch("utils.abc_utils.time.sleep", return_value=None)
@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_persistent_5xx_retries_then_raises(mock_request, mock_urlopen, _tok, _sleep):
    mock_urlopen.side_effect = HTTPError(
        url="http://x/topic_entity_tag/", code=500, msg="Server Error", hdrs=None, fp=None)

    with pytest.raises(RuntimeError):
        abc_utils.send_classification_tag_to_abc(
            reference_curie="AGRKB:101000000000001",
            species="NCBITaxon:6239",
            topic="ATP:0000096",
            negated=False,
            data_novelty="ATP:0000335",
            confidence_score=None,
            confidence_level=None,
            tet_source_id=42,
        )

    assert mock_urlopen.call_count == 3
