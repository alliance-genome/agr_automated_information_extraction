"""FB "no genetic data" classification (job and model topic ATP:0000207).

SCRUM-6669 / SCRUM-6670: every paper gets a "genetic data" (ATP:0000402) topic tag -- negated
for a positive call ("no genetic data"), non-negated with confidence level NEG for a negative call.
No manual_indexing_tag is written any more.
SCRUM-6203: the "manual indexing status TBD" workflow tag (ATP:0000359) is only added when the
confidence is >= 0.5 and the paper has no manual indexing status yet.
"""
from unittest.mock import patch

from agr_document_classifier import agr_document_classifier_classify as clf


def _run(conf_scores, classifications, current_status=None, tag_result=True, species="NCBITaxon:7214"):
    curies = [f"AGRKB:10100000000000{i}" for i in range(len(conf_scores))]
    files_loaded = [f"/tmp/{c.replace(':', '_')}.md" for c in curies]
    valid_embeddings = [True] * len(conf_scores)
    job_map = {c: {"job": i} for i, c in enumerate(curies)}
    model_meta_data = {"negated": True, "data_novelty": "ATP:0000335", "data_context": "ATP:0000405",
                       "species": species, "ml_model_id": 60}
    with patch.object(clf, "send_classification_tag_to_abc", return_value=tag_result) as m_tet, \
            patch.object(clf, "get_current_workflow_status", return_value=current_status), \
            patch.object(clf, "create_workflow_tag") as m_wft, \
            patch.object(clf, "set_job_success") as m_success, \
            patch.object(clf, "set_job_started"), \
            patch.object(clf, "set_job_failure"), \
            patch.object(clf, "get_cached_mod_species_map", return_value={"FB": "NCBITaxon:7227"}):
        clf.send_classification_results(
            files_loaded, classifications, conf_scores, valid_embeddings, job_map,
            "FB", "ATP:0000207", 1, model_meta_data)
    return curies, m_tet, m_wft, m_success


def test_positive_call_is_a_negated_genetic_data_tag():
    curies, m_tet, _, _ = _run(conf_scores=[0.9], classifications=[1])
    assert m_tet.call_count == 1
    args, kwargs = m_tet.call_args
    assert args == (curies[0], "NCBITaxon:7214", "ATP:0000402")
    assert kwargs["negated"] is True
    assert kwargs["confidence_score"] == 0.9
    assert kwargs["confidence_level"] == "HIGH"
    assert kwargs["data_novelty"] == "ATP:0000335"
    assert kwargs["data_context"] == "ATP:0000405"
    assert kwargs["tet_source_id"] == 1
    assert kwargs["ml_model_id"] == 60


def test_negative_call_is_a_non_negated_genetic_data_tag_with_level_neg():
    # SCRUM-6670: the paper is kept, as the old manual_indexing_tag did, with the model's
    # probability of "no genetic data" as its score; no TBD workflow tag
    curies, m_tet, m_wft, m_success = _run(conf_scores=[0.3], classifications=[0])
    assert m_tet.call_count == 1
    args, kwargs = m_tet.call_args
    assert args == (curies[0], "NCBITaxon:7214", "ATP:0000402")
    assert kwargs["negated"] is False
    assert kwargs["confidence_score"] == 0.3
    assert kwargs["confidence_level"] == "NEG"
    assert kwargs["ml_model_id"] == 60
    assert m_wft.call_count == 0
    assert m_success.call_count == 1


def test_tbd_tag_only_for_confidence_at_or_above_half():
    curies, m_tet, m_wft, _ = _run(conf_scores=[0.9, 0.3], classifications=[1, 0])
    assert m_tet.call_count == 2
    assert m_wft.call_count == 1
    _, kwargs = m_wft.call_args
    assert kwargs["reference_curie"] == curies[0]
    assert kwargs["workflow_tag_atp_id"] == "ATP:0000359"


def test_tbd_tag_created_at_exactly_half():
    # 0.5 counts as positive ("0.5 or more")
    _, m_tet, m_wft, _ = _run(conf_scores=[0.5], classifications=[1])
    assert m_tet.call_count == 1
    assert m_wft.call_count == 1


def test_no_tbd_tag_below_half_even_for_a_positive_call():
    _, m_tet, m_wft, _ = _run(conf_scores=[0.49], classifications=[1])
    assert m_tet.call_count == 1
    assert m_wft.call_count == 0


def test_no_tbd_tag_when_a_manual_indexing_status_exists():
    _, m_tet, m_wft, _ = _run(conf_scores=[0.9], classifications=[1], current_status="ATP:0000343")
    assert m_tet.call_count == 1
    assert m_wft.call_count == 0


def test_failed_tag_write_leaves_the_job_open_and_adds_no_tbd_tag():
    _, m_tet, m_wft, m_success = _run(conf_scores=[0.9], classifications=[1], tag_result=False)
    assert m_tet.call_count == 1
    assert m_wft.call_count == 0
    assert m_success.call_count == 0


def test_model_without_a_species_gets_the_mod_default_taxon():
    curies, m_tet, _, _ = _run(conf_scores=[0.9], classifications=[1], species=None)
    args, _ = m_tet.call_args
    assert args == (curies[0], "NCBITaxon:7227", "ATP:0000402")


def test_failed_negative_tag_write_leaves_the_job_open():
    _, m_tet, m_wft, m_success = _run(conf_scores=[0.2], classifications=[0], tag_result=False)
    assert m_tet.call_count == 1
    assert m_wft.call_count == 0
    assert m_success.call_count == 0
