"""SCRUM-5697: data_context threading through the two TET-creating helpers.

The field is optional and omitted from the payload when the model carries no
value, rather than sent as an explicit null -- the ABC applies its own default,
and a null would fail the schema's min_length constraint.
"""
import json
from unittest.mock import MagicMock, patch

from utils import abc_utils


def _read_payload_from_request(mock_request_cls):
    """Pull the JSON body out of the urlopen Request object the SUT built."""
    args, kwargs = mock_request_cls.call_args
    data = kwargs.get("data") or args[1]
    return json.loads(data.decode("utf-8"))


def _classification_kwargs(**overrides):
    kwargs = {
        "reference_curie": "AGRKB:101000000000001",
        "species": "NCBITaxon:6239",
        "topic": "ATP:0000096",
        "negated": False,
        "data_novelty": "ATP:0000335",
        "confidence_score": None,
        "confidence_level": None,
        "tet_source_id": 42,
    }
    kwargs.update(overrides)
    return kwargs


def _entity_kwargs(**overrides):
    kwargs = {
        "reference_curie": "AGRKB:101000000000001",
        "species": "NCBITaxon:6239",
        "data_novelty": "ATP:0000334",
        "topic": "ATP:0000005",
        "tet_source_id": 42,
        "entity": "WB:WBGene00003001",
        "entity_type": "ATP:0000005",
    }
    kwargs.update(overrides)
    return kwargs


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_classification_data_context_included_when_provided(mock_request, mock_urlopen, _tok):
    mock_urlopen.return_value.__enter__.return_value.getcode.return_value = 201

    abc_utils.send_classification_tag_to_abc(
        **_classification_kwargs(data_context="ATP:0000405"))

    payload = _read_payload_from_request(mock_request)
    assert payload["data_context"] == "ATP:0000405"


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_classification_data_context_omitted_when_none(mock_request, mock_urlopen, _tok):
    mock_urlopen.return_value.__enter__.return_value.getcode.return_value = 201

    abc_utils.send_classification_tag_to_abc(**_classification_kwargs(data_context=None))

    payload = _read_payload_from_request(mock_request)
    assert "data_context" not in payload


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_entity_data_context_included_when_provided(mock_request, mock_urlopen, _tok):
    mock_urlopen.return_value.__enter__.return_value.getcode.return_value = 201

    abc_utils.send_entity_tag_to_abc(**_entity_kwargs(data_context="ATP:0000360"))

    payload = _read_payload_from_request(mock_request)
    assert payload["data_context"] == "ATP:0000360"


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_entity_data_context_omitted_when_none(mock_request, mock_urlopen, _tok):
    mock_urlopen.return_value.__enter__.return_value.getcode.return_value = 201

    abc_utils.send_entity_tag_to_abc(**_entity_kwargs(data_context=None))

    payload = _read_payload_from_request(mock_request)
    assert "data_context" not in payload
    # The other fields still go out unchanged.
    assert payload["data_novelty"] == "ATP:0000334"
    assert payload["entity"] == "WB:WBGene00003001"


def test_upload_ml_model_sends_data_context(tmp_path):
    """The value lives on the ml_model row, so the trainer's upload must carry it.

    upload_ml_model posts through ``requests``, not urllib, and sends the metadata
    as multipart form fields -- so mock requests.post and assert on ``data``.
    """
    model_path = tmp_path / "WB_ATP_0000082_classifier.joblib"
    model_path.write_bytes(b"model-bytes")
    stats = {"model_name": "logreg", "average_precision": 0.9, "average_recall": 0.8,
             "average_f1": 0.85, "best_params": None}
    resp = MagicMock(status_code=201)

    with patch("utils.abc_utils.get_authentication_token", return_value="t"), \
            patch("utils.abc_utils.generate_headers",
                  return_value={"Content-Type": "application/json"}), \
            patch("utils.abc_utils.requests.post", return_value=resp) as mock_post:
        abc_utils.upload_ml_model(
            task_type="biocuration_topic_classification",
            mod_abbreviation="WB",
            topic="ATP:0000082",
            model_path=str(model_path),
            stats=stats,
            file_extension="joblib",
            data_novelty="ATP:0000335",
            data_context="ATP:0000405",
        )

    assert mock_post.call_args.kwargs["data"]["data_context"] == "ATP:0000405"
    assert mock_post.call_args.kwargs["data"]["data_novelty"] == "ATP:0000335"


def test_upload_ml_model_data_context_defaults_to_none(tmp_path):
    """Omitting it leaves the ml_model column NULL rather than guessing a term."""
    model_path = tmp_path / "WB_ATP_0000082_classifier.joblib"
    model_path.write_bytes(b"model-bytes")
    stats = {"model_name": "logreg", "average_precision": 0.9, "average_recall": 0.8,
             "average_f1": 0.85, "best_params": None}
    resp = MagicMock(status_code=201)

    with patch("utils.abc_utils.get_authentication_token", return_value="t"), \
            patch("utils.abc_utils.generate_headers",
                  return_value={"Content-Type": "application/json"}), \
            patch("utils.abc_utils.requests.post", return_value=resp) as mock_post:
        abc_utils.upload_ml_model(
            task_type="biocuration_topic_classification",
            mod_abbreviation="WB",
            topic="ATP:0000082",
            model_path=str(model_path),
            stats=stats,
            file_extension="joblib",
            data_novelty="ATP:0000335",
        )

    assert mock_post.call_args.kwargs["data"]["data_context"] is None


class TestDefaultDataContextForMod:
    """SCRUM-5697 / SCRUM-6815. The default data_context is per MOD and per shape
    (topic-only vs with an entity), from DATA_CONTEXT_DEFAULTS -- the same table as
    the ABC. ATP:0000325 was deprecated and every MOD moved off it in one cutover.
    """

    def test_wb_topic_tags_take_experimental_data(self):
        # Kimberly Van Auken on SCRUM-6815 (2026-11-10): WB matches ZFIN
        assert abc_utils.default_data_context_for_mod("WB") == "ATP:0000405"

    def test_wb_entities_are_experimentally_studied_entities(self):
        assert abc_utils.default_data_context_for_mod("WB", has_entity=True) == "ATP:0000406"

    def test_fb_takes_experimental_data_whatever_the_shape(self):
        assert abc_utils.default_data_context_for_mod("FB") == "ATP:0000405"
        assert abc_utils.default_data_context_for_mod("FB", has_entity=True) == "ATP:0000405"

    def test_zfin_splits_by_shape(self):
        assert abc_utils.default_data_context_for_mod("ZFIN") == "ATP:0000405"
        assert abc_utils.default_data_context_for_mod("ZFIN", has_entity=True) == "ATP:0000406"

    def test_other_mods_take_experimental_data(self):
        for mod in ("MGI", "SGD", "RGD", "XB"):
            assert abc_utils.default_data_context_for_mod(mod) == "ATP:0000405"
            assert abc_utils.default_data_context_for_mod(mod, has_entity=True) == "ATP:0000405"

    def test_an_unresolved_mod_takes_the_server_default(self):
        """--mod_train is not required, so it can be absent; the ABC falls back to
        the same value."""
        assert abc_utils.default_data_context_for_mod(None) == "ATP:0000405"
        assert abc_utils.default_data_context_for_mod("") == "ATP:0000405"

    def test_nothing_defaults_to_the_deprecated_term(self):
        for mod in ("WB", "FB", "ZFIN", "MGI", "SGD", "RGD", "XB", None):
            for has_entity in (False, True):
                assert abc_utils.default_data_context_for_mod(mod, has_entity) != "ATP:0000325"
