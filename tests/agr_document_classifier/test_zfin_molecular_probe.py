"""SCRUM-5764: ZFIN "molecular probe" (ATP:0000370) classification results.

A paper the abstract classifier calls a probe paper with at least
ZFIN_MOLECULAR_PROBE_THRESHOLD confidence is dropped from the ZFIN acquisition
workflow, which means three writes:

  * the ATP:0000370 topic tag (never negated -- only positives are reported)
  * the "won't manually index" (ATP:0000343) workflow tag
  * a "won't curate" (ATP:0000299) curation_status row on the whole paper
    (ATP:0000002)

with the last two carrying the "outside of scope" (ATP:0000209) curation tag.

Below the threshold nothing at all is written: Ceri chose (SCRUM-5764,
2026-08-17) to let borderline papers reach a curator untouched, because wrongly
dropping a curatable paper is the expensive mistake -- nobody ever sees it
again.
"""
from unittest.mock import patch

from agr_document_classifier import agr_document_classifier_classify as clf


def _run(conf_scores, classifications, negated_model=False, mod_abbr="ZFIN",
         topic="ATP:0000370"):
    curies = [f"AGRKB:10100000000000{i}" for i in range(len(conf_scores))]
    files_loaded = [f"/tmp/{c.replace(':', '_')}.md" for c in curies]
    valid_embeddings = [True] * len(conf_scores)
    job_map = {c: {"job": i} for i, c in enumerate(curies)}
    model_meta_data = {"negated": negated_model, "data_novelty": "ATP:0000335",
                       "species": "NCBITaxon:7955", "ml_model_id": 71}
    with patch.object(clf, "send_classification_tag_to_abc", return_value=True) as m_tet, \
            patch.object(clf, "create_workflow_tag", return_value=True) as m_wft, \
            patch.object(clf, "send_curation_status_to_abc", return_value=True) as m_cs, \
            patch.object(clf, "send_manual_indexing_to_abc") as m_mi, \
            patch.object(clf, "set_job_success") as m_success, \
            patch.object(clf, "set_job_started"), \
            patch.object(clf, "set_job_failure"):
        clf.send_classification_results(
            files_loaded, classifications, conf_scores, valid_embeddings, job_map,
            mod_abbr, topic, 1, model_meta_data)
    return curies, m_tet, m_wft, m_cs, m_mi, m_success


def test_confident_probe_paper_gets_topic_tag_wont_index_and_wont_curate():
    curies, m_tet, m_wft, m_cs, _, _ = _run(conf_scores=[0.95], classifications=[1])

    _, tet_kwargs = m_tet.call_args
    assert m_tet.call_args[0][0] == curies[0]
    assert tet_kwargs["negated"] is False

    _, wft_kwargs = m_wft.call_args
    assert wft_kwargs == {"reference_curie": curies[0], "mod_abbreviation": "ZFIN",
                          "workflow_tag_atp_id": "ATP:0000343",
                          "curation_tag": "ATP:0000209"}

    _, cs_kwargs = m_cs.call_args
    assert cs_kwargs == {"reference_curie": curies[0], "mod_abbreviation": "ZFIN",
                         "topic": "ATP:0000002",
                         "curation_status": "ATP:0000299",
                         "curation_tag": "ATP:0000209"}


def test_topic_tag_uses_the_probe_topic_and_the_model_metadata():
    curies, m_tet, _, _, _, _ = _run(conf_scores=[0.95], classifications=[1])

    args, kwargs = m_tet.call_args
    assert args[2] == "ATP:0000370"
    assert kwargs["confidence_score"] == 0.95
    assert kwargs["confidence_level"] == "HIGH"
    assert kwargs["ml_model_id"] == 71


def test_paper_at_exactly_the_threshold_is_dropped():
    _, m_tet, m_wft, m_cs, _, _ = _run(conf_scores=[0.87], classifications=[1])
    assert m_tet.call_count == 1
    assert m_wft.call_count == 1
    assert m_cs.call_count == 1


def test_positive_below_the_threshold_writes_nothing_but_completes_the_job():
    _, m_tet, m_wft, m_cs, _, m_success = _run(conf_scores=[0.86], classifications=[1])
    assert m_tet.call_count == 0
    assert m_wft.call_count == 0
    assert m_cs.call_count == 0
    # the reference was classified successfully -- there is just nothing to report
    assert m_success.call_count == 1


def test_negatives_never_produce_a_negated_tag_even_for_a_negated_model():
    # a negated model would normally get a negated=True TET for every negative
    _, m_tet, m_wft, m_cs, _, m_success = _run(
        conf_scores=[0.02], classifications=[0], negated_model=True)
    assert m_tet.call_count == 0
    assert m_wft.call_count == 0
    assert m_cs.call_count == 0
    assert m_success.call_count == 1


def test_only_confident_papers_in_a_mixed_batch_are_dropped():
    curies, m_tet, m_wft, m_cs, _, m_success = _run(
        conf_scores=[0.99, 0.5, 0.01], classifications=[1, 1, 0])
    assert m_tet.call_count == 1
    assert m_wft.call_count == 1
    assert m_cs.call_count == 1
    assert m_wft.call_args[1]["reference_curie"] == curies[0]
    # every job in the batch still completes
    assert m_success.call_count == 3


def test_the_drop_is_skipped_when_the_topic_tag_write_fails():
    """No won't-index/won't-curate on a reference whose topic tag never landed."""
    curies = ["AGRKB:101000000000001"]
    job_map = {curies[0]: {"job": 1}}
    model_meta_data = {"negated": False, "data_novelty": "ATP:0000335",
                       "species": "NCBITaxon:7955", "ml_model_id": 71}
    with patch.object(clf, "send_classification_tag_to_abc", return_value=False), \
            patch.object(clf, "create_workflow_tag") as m_wft, \
            patch.object(clf, "send_curation_status_to_abc") as m_cs, \
            patch.object(clf, "set_job_success") as m_success, \
            patch.object(clf, "set_job_started"), \
            patch.object(clf, "set_job_failure"):
        clf.send_classification_results(
            ["/tmp/AGRKB_101000000000001.md"], [1], [0.99], [True], job_map,
            "ZFIN", "ATP:0000370", 1, model_meta_data)
    assert m_wft.call_count == 0
    assert m_cs.call_count == 0
    assert m_success.call_count == 0


def test_other_zfin_topics_keep_the_standard_tagging_path():
    _, m_tet, m_wft, m_cs, m_mi, _ = _run(
        conf_scores=[0.6], classifications=[1], topic="ATP:0000009")
    # standard path: topic tag only, no acquisition-workflow drop
    assert m_tet.call_count == 1
    assert m_wft.call_count == 0
    assert m_cs.call_count == 0
    assert m_mi.call_count == 0


def test_job_still_completes_when_a_drop_write_fails():
    """A failed drop write leaves the paper in the ZFIN workflow -- the safe
    direction, since a curator then still sees it -- so the job is not retried.
    Retrying would re-POST the workflow tag and get a permanent 422 duplicate.
    """
    curies = ["AGRKB:101000000000001"]
    job_map = {curies[0]: {"job": 1}}
    model_meta_data = {"negated": False, "data_novelty": "ATP:0000335",
                       "species": "NCBITaxon:7955", "ml_model_id": 71}
    with patch.object(clf, "send_classification_tag_to_abc", return_value=True), \
            patch.object(clf, "create_workflow_tag", return_value=False), \
            patch.object(clf, "send_curation_status_to_abc", return_value=True) as m_cs, \
            patch.object(clf, "set_job_success") as m_success, \
            patch.object(clf, "set_job_started"), \
            patch.object(clf, "set_job_failure"):
        clf.send_classification_results(
            ["/tmp/AGRKB_101000000000001.md"], [1], [0.99], [True], job_map,
            "ZFIN", "ATP:0000370", 1, model_meta_data)
    # the curation status is still attempted after the workflow tag failed
    assert m_cs.call_count == 1
    assert m_success.call_count == 1


def test_a_pre_existing_curation_status_is_not_reported_as_an_error(caplog):
    """send_curation_status_to_abc returns False both for a real HTTP failure and
    for "a status already exists here, left untouched" -- and it has already logged
    each at the right level. The caller must not flatten both back to ERROR.
    """
    job_map = {"AGRKB:101000000000001": {"job": 1}}
    model_meta_data = {"negated": False, "data_novelty": "ATP:0000335",
                       "species": "NCBITaxon:7955", "ml_model_id": 71}
    with patch.object(clf, "send_classification_tag_to_abc", return_value=True), \
            patch.object(clf, "create_workflow_tag", return_value=True), \
            patch.object(clf, "send_curation_status_to_abc", return_value=False), \
            patch.object(clf, "set_job_success"), \
            patch.object(clf, "set_job_started"), \
            patch.object(clf, "set_job_failure"):
        with caplog.at_level("DEBUG", logger=clf.__name__):
            clf.send_classification_results(
                ["/tmp/AGRKB_101000000000001.md"], [1], [0.99], [True], job_map,
                "ZFIN", "ATP:0000370", 1, model_meta_data)

    assert not [r for r in caplog.records if r.levelname == "ERROR"]
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "not fully dropped" in warnings[0].message
