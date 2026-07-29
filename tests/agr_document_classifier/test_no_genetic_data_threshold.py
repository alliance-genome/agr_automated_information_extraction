"""SCRUM-6203: FB "no genetic data" (ATP:0000207) papers only get the
"manual indexing status TBD" workflow tag (ATP:0000359) when the classifier's
positive-class confidence is >= 0.5. A manual_indexing_tag is still written for
every paper regardless of confidence (the weekly FB export consumes all scores).
"""
from unittest.mock import patch

from agr_document_classifier import agr_document_classifier_classify as clf


def _run(conf_scores, classifications):
    curies = [f"AGRKB:10100000000000{i}" for i in range(len(conf_scores))]
    files_loaded = [f"/tmp/{c.replace(':', '_')}.md" for c in curies]
    valid_embeddings = [True] * len(conf_scores)
    job_map = {c: {"job": i} for i, c in enumerate(curies)}
    model_meta_data = {"negated": True, "data_novelty": "ATP:0000321",
                       "species": None, "ml_model_id": 67}
    with patch.object(clf, "send_manual_indexing_to_abc", return_value=True) as m_mi, \
            patch.object(clf, "get_current_workflow_status", return_value=None), \
            patch.object(clf, "create_workflow_tag") as m_ct, \
            patch.object(clf, "set_job_success"), \
            patch.object(clf, "set_job_started"), \
            patch.object(clf, "set_job_failure"), \
            patch.object(clf, "send_classification_tag_to_abc"):
        clf.send_classification_results(
            files_loaded, classifications, conf_scores, valid_embeddings, job_map,
            "FB", "ATP:0000207", 1, model_meta_data)
    return curies, m_mi, m_ct


def test_tbd_tag_only_for_confidence_at_or_above_half():
    curies, m_mi, m_ct = _run(conf_scores=[0.9, 0.3], classifications=[1, 0])
    # manual_indexing_tag is written for EVERY paper (export needs all scores)
    assert m_mi.call_count == 2
    # TBD workflow tag created only for the >= 0.5 paper
    assert m_ct.call_count == 1
    _, kwargs = m_ct.call_args
    assert kwargs["reference_curie"] == curies[0]
    assert kwargs["workflow_tag_atp_id"] == "ATP:0000359"


def test_tbd_tag_created_at_exactly_half():
    # 0.5 counts as positive ("0.5 or more")
    _, m_mi, m_ct = _run(conf_scores=[0.5], classifications=[1])
    assert m_mi.call_count == 1
    assert m_ct.call_count == 1


def test_no_tbd_tag_below_half():
    _, m_mi, m_ct = _run(conf_scores=[0.49], classifications=[0])
    # still recorded for the export, but not surfaced for manual indexing
    assert m_mi.call_count == 1
    assert m_ct.call_count == 0
