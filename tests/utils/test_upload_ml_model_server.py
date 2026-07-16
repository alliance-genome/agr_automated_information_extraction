import os
import tempfile
from unittest.mock import MagicMock, patch

from utils import abc_utils

_STATS = {"model_name": "XGBClassifier", "average_precision": 0.9,
          "average_recall": 0.8, "average_f1": 0.85, "best_params": {"a": 1}}


def _run_upload(monkeypatch_env):
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "FB_ATP_0000207_classifier.joblib")
        with open(model_path, "wb") as handle:
            handle.write(b"model-bytes")
        resp = MagicMock(status_code=201)
        with patch("utils.abc_utils.get_authentication_token", return_value="t"), \
                patch("utils.abc_utils.generate_headers", return_value={"Content-Type": "application/json"}), \
                patch("utils.abc_utils.requests.post", return_value=resp) as mock_post:
            for key, value in monkeypatch_env.items():
                os.environ[key] = value
            try:
                abc_utils.upload_ml_model(
                    task_type="biocuration_topic_classification", mod_abbreviation="FB",
                    topic="ATP:0000207", model_path=model_path, stats=_STATS, dataset_id=39,
                    file_extension="joblib", production=False, description="[abc_embeddings]")
            finally:
                for key in monkeypatch_env:
                    os.environ.pop(key, None)
        return mock_post.call_args[0][0], mock_post.call_args


def test_upload_url_uses_override_when_set():
    url, call = _run_upload({"ABC_UPLOAD_API_SERVER": "https://stage-literature-rest.alliancegenome.org"})
    assert url == "https://stage-literature-rest.alliancegenome.org/ml_model/upload"
    # description carrying the marker is sent in the form data
    assert call.kwargs["data"]["description"] == "[abc_embeddings]"


def test_upload_url_defaults_to_blue_api_when_unset():
    url, _ = _run_upload({})
    assert url == f"{abc_utils.blue_api_base_url}/ml_model/upload"
