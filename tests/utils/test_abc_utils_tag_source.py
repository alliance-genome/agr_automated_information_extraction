"""SCRUM-6518: the ABC renamed topic_entity_tag_source to tag_source.

The lookup/create endpoint moved from /topic_entity_tag/source to /tag_source,
and the id field on both the source objects and the topic_entity_tag payload is
now tag_source_id. The old endpoint 404s and the old payload key is rejected by
the strict TET schema, which took the production classifier down on 2026-09-23.
"""
import json
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from utils import abc_utils


def _request_url(mock_request_cls, call_index=0):
    args, kwargs = mock_request_cls.call_args_list[call_index]
    return kwargs.get("url") or args[0]


def _request_payload(mock_request_cls, call_index=0):
    args, kwargs = mock_request_cls.call_args_list[call_index]
    data = kwargs.get("data") or args[1]
    return json.loads(data.decode("utf-8"))


def _response_returning(body: dict):
    ctx = MagicMock()
    ctx.__enter__.return_value.read.return_value = json.dumps(body).encode("utf-8")
    ctx.__enter__.return_value.getcode.return_value = 201
    return ctx


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_get_tet_source_id_reads_the_tag_source_endpoint(mock_request, mock_urlopen, _tok):
    mock_urlopen.return_value = _response_returning({"tag_source_id": 176, "source_method": "abc_document_classifier"})

    source_id = abc_utils.get_tet_source_id("ZFIN", "abc_document_classifier", "desc")

    assert source_id == 176
    url = _request_url(mock_request)
    assert url.endswith("/tag_source/ECO:0008004/abc_document_classifier/ZFIN/ZFIN")
    assert "/topic_entity_tag/source" not in url


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_get_tet_source_id_creates_a_missing_source_through_tag_source(mock_request, mock_urlopen, _tok):
    not_found = HTTPError(url="u", code=404, msg="Not Found", hdrs=None, fp=None)
    mock_urlopen.side_effect = [not_found, _response_returning({"tag_source_id": 240})]

    source_id = abc_utils.get_tet_source_id("AGR", "abc_entity_extractor", "desc")

    assert source_id == 240
    create_url = _request_url(mock_request, call_index=1)
    assert create_url.endswith("/tag_source")
    assert "/topic_entity_tag/source" not in create_url
    created = _request_payload(mock_request, call_index=1)
    assert created["source_evidence_assertion"] == "ECO:0008021"
    assert created["data_provider"] == "AGR"
    assert created["secondary_data_provider_abbreviation"] == "AGR"


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_classification_tag_payload_uses_tag_source_id(mock_request, mock_urlopen, _tok):
    mock_urlopen.return_value.__enter__.return_value.getcode.return_value = 201

    assert abc_utils.send_classification_tag_to_abc(
        reference_curie="AGRKB:101000000000001", species="NCBITaxon:7955", topic="ATP:0000370",
        negated=False, data_novelty="ATP:0000335", confidence_score=0.99, confidence_level="HIGH",
        tet_source_id=176)

    payload = _request_payload(mock_request)
    assert payload["tag_source_id"] == 176
    assert "topic_entity_tag_source_id" not in payload


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_entity_tag_payload_uses_tag_source_id(mock_request, mock_urlopen, _tok):
    mock_urlopen.return_value.__enter__.return_value.getcode.return_value = 201

    assert abc_utils.send_entity_tag_to_abc(
        reference_curie="AGRKB:101000000000001", species="NCBITaxon:6239", data_novelty="ATP:0000335",
        topic="ATP:0000005", tet_source_id=155, entity="WB:WBGene00000001", entity_type="ATP:0000005")

    payload = _request_payload(mock_request)
    assert payload["tag_source_id"] == 155
    assert "topic_entity_tag_source_id" not in payload
