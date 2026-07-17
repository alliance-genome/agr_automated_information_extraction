from unittest.mock import MagicMock, patch

import numpy as np

from agr_document_classifier import agr_document_classifier_reclassify as rc
from agr_document_classifier import agr_document_classifier_classify as cls


@patch("agr_document_classifier.agr_document_classifier_reclassify.psycopg2.connect")
def test_update_manual_indexing_confidence_updates_prediction_only(mock_connect):
    conn = MagicMock()
    cur = MagicMock()
    mock_connect.return_value = conn
    conn.cursor.return_value = cur
    cur.fetchone.return_value = (7,)   # mod_id lookup
    cur.rowcount = 1

    n = rc.update_manual_indexing_confidence(
        {"AGRKB:1": 0.9, "AGRKB:2": 0.1}, "FB", "ATP:0000207")

    assert n == 2
    conn.commit.assert_called_once()
    update_sqls = [c.args[0] for c in cur.execute.call_args_list if "UPDATE" in c.args[0]]
    assert len(update_sqls) == 2
    for sql in update_sqls:
        # Only the prediction column is written; the curator column is never touched.
        assert "confidence_score" in sql
        assert "validation_by_biocurator" not in sql


def test_update_manual_indexing_confidence_empty_is_noop():
    # No DB connection attempted, returns 0.
    assert rc.update_manual_indexing_confidence({}, "FB", "ATP:0000207") == 0


def test_topic_config():
    # The 4 TET topics + the separate no-genetic-data topic.
    assert rc.NO_GEN_DATA_TOPIC == "ATP:0000207"
    assert set(rc.TET_TOPICS) == {"ATP:0000152", "ATP:0000069", "ATP:0000006", "ATP:0000013"}
    assert rc.NO_GEN_DATA_TOPIC not in rc.TET_TOPICS


@patch("agr_document_classifier.agr_document_classifier_reclassify.psycopg2.connect")
def test_reference_classification_complete_query(mock_connect):
    conn = MagicMock()
    cur = MagicMock()
    mock_connect.return_value = conn
    conn.cursor.return_value = cur
    cur.fetchall.return_value = [("AGRKB:1",), ("AGRKB:2",)]

    out = rc.get_reference_classification_complete_curies("FB")

    assert out == ["AGRKB:1", "AGRKB:2"]
    sql, params = cur.execute.call_args.args
    assert "workflow_tag" in sql
    # Enumerates the union of the per-data-type "classification complete" tags.
    assert params == ("FB", rc.REFERENCE_CLASSIFICATION_COMPLETE_TAGS)


@patch("agr_document_classifier.agr_document_classifier_reclassify.send_classification_tag_to_abc")
@patch("agr_document_classifier.agr_document_classifier_reclassify.classify_documents_from_abc_embeddings")
@patch("agr_document_classifier.agr_document_classifier_reclassify._load_model")
def test_tet_gate_negated_model_tags_every_reference(mock_load, mock_classify, mock_send):
    mock_load.return_value = (object(), {"species": None, "negated": True,
                                         "data_novelty": "ATP:0000335", "ml_model_id": 68})
    mock_classify.return_value = (["c1", "c2", "c3"], [1, 0, 0], [0.9, 0.1, 0.2], [True, True, True])
    mock_send.return_value = True

    res = rc.reclassify_tet_topic("ATP:0000152", "disease", ["c1", "c2", "c3"], 5, {}, dry_run=False)

    assert res["tets"] == 3           # negated model tags positives AND negatives
    assert res["not_tagged"] == 0
    assert mock_send.call_count == 3


@patch("agr_document_classifier.agr_document_classifier_reclassify.send_classification_tag_to_abc")
@patch("agr_document_classifier.agr_document_classifier_reclassify.classify_documents_from_abc_embeddings")
@patch("agr_document_classifier.agr_document_classifier_reclassify._load_model")
def test_tet_gate_non_negated_model_tags_only_positives(mock_load, mock_classify, mock_send):
    mock_load.return_value = (object(), {"species": None, "negated": False,
                                         "data_novelty": "ATP:0000335", "ml_model_id": 99})
    mock_classify.return_value = (["c1", "c2", "c3"], [1, 0, 1], [0.9, 0.2, 0.8], [True, True, True])
    mock_send.return_value = True

    res = rc.reclassify_tet_topic("ATP:0000152", "disease", ["c1", "c2", "c3"], 5, {}, dry_run=False)

    assert res["tets"] == 2           # matches production gate: negatives dropped
    assert res["not_tagged"] == 1
    assert mock_send.call_count == 2


@patch("agr_document_classifier.agr_document_classifier_classify.predict_labels_and_confidence")
@patch("agr_document_classifier.agr_document_classifier_classify.get_reference_embedding")
def test_embedding_cache_fetches_each_reference_once(mock_get_emb, mock_predict):
    mock_get_emb.return_value = (np.zeros(cls.ABC_EMBEDDING_DIM, dtype=np.float32), "text")
    mock_predict.return_value = (np.array([1]), [0.9])
    cache: dict = {}

    # Same reference classified by two different models sharing one cache.
    cls.classify_documents_from_abc_embeddings(["AGRKB:1"], "FB", object(), embedding_cache=cache)
    cls.classify_documents_from_abc_embeddings(["AGRKB:1"], "FB", object(), embedding_cache=cache)

    assert mock_get_emb.call_count == 1
    assert "AGRKB:1" in cache
