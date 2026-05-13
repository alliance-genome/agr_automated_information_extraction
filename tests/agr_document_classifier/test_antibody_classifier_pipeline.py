"""Integration tests for the antibody string-matching classifier pipeline.

Mocks all ABC API calls and the Markdown loader. Exercises the per-job logic
end-to-end: MD text -> rule matches -> TET POST payload + workflow status calls.
"""

from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def patched_pipeline():
    """Patch every external dependency the pipeline talks to."""
    p = "agr_document_classifier.agr_antibody_string_matching_classifier"
    with patch(f"{p}.get_all_curated_entities") as m_genes, \
         patch(f"{p}.get_tet_source_id", return_value=42), \
         patch(f"{p}.download_md_files_for_references") as m_download, \
         patch(f"{p}.send_classification_tag_to_abc", return_value=True) as m_send, \
         patch(f"{p}.set_job_started", return_value=True) as m_started, \
         patch(f"{p}.set_job_success", return_value=True) as m_success, \
         patch(f"{p}.set_job_failure", return_value=True) as m_failure, \
         patch(f"{p}.AllianceMarkdown") as m_md_cls, \
         patch(f"{p}.os.listdir", return_value=["AGRKB_101000000000001.md"]), \
         patch(f"{p}.os.remove"), \
         patch(f"{p}.os.makedirs"):
        m_genes.return_value = (["pdr-1", "unc-54"], {}, {})
        yield {
            "genes": m_genes,
            "download": m_download,
            "send": m_send,
            "started": m_started,
            "success": m_success,
            "failure": m_failure,
            "md_cls": m_md_cls,
        }


def _job(curie="AGRKB:101000000000001"):
    return {"reference_curie": curie, "reference_workflow_tag_id": 999, "mod_id": 3}


def _set_md_text(mocks, sentences):
    inst = mocks["md_cls"].return_value
    inst.load_from_file = MagicMock()
    inst.get_sentences = MagicMock(return_value=sentences)


def test_positive_paper_emits_tet_with_sorted_note(patched_pipeline):
    from agr_document_classifier import agr_antibody_string_matching_classifier as pipe
    _set_md_text(patched_pipeline, [
        "The antibody was raised against UNC-54.",
        "We also used anti-PDR-1 in some experiments.",
    ])
    pipe.process_antibody_jobs(topic="ATP:0000096", jobs=[_job()])

    assert patched_pipeline["send"].call_count == 1
    kwargs = patched_pipeline["send"].call_args.kwargs
    assert kwargs["reference_curie"] == "AGRKB:101000000000001"
    assert kwargs["topic"] == "ATP:0000096"
    assert kwargs["species"] == "NCBITaxon:6239"
    assert kwargs["negated"] is False
    assert kwargs["data_novelty"] == "ATP:0000335"
    assert kwargs["confidence_score"] is None
    assert kwargs["confidence_level"] is None
    assert kwargs["note"] == "anti-PDR-1, raised antibody"

    patched_pipeline["started"].assert_called_once()
    patched_pipeline["success"].assert_called_once()
    patched_pipeline["failure"].assert_not_called()


def test_negative_paper_emits_negated_tet_without_note(patched_pipeline):
    from agr_document_classifier import agr_antibody_string_matching_classifier as pipe
    _set_md_text(patched_pipeline, [
        "We bought a commercial reagent from Sigma.",
        "The animals were maintained at 20 degrees.",
    ])
    pipe.process_antibody_jobs(topic="ATP:0000096", jobs=[_job()])

    kwargs = patched_pipeline["send"].call_args.kwargs
    assert kwargs["negated"] is True
    assert kwargs["note"] is None
    patched_pipeline["success"].assert_called_once()


def test_md_parse_failure_marks_job_failed(patched_pipeline):
    from agr_document_classifier import agr_antibody_string_matching_classifier as pipe
    patched_pipeline["md_cls"].return_value.load_from_file.side_effect = ValueError("bad MD")
    pipe.process_antibody_jobs(topic="ATP:0000096", jobs=[_job()])

    patched_pipeline["failure"].assert_called_once()
    patched_pipeline["send"].assert_not_called()
    patched_pipeline["success"].assert_not_called()


def test_empty_curated_gene_list_aborts(patched_pipeline):
    from agr_document_classifier import agr_antibody_string_matching_classifier as pipe
    patched_pipeline["genes"].return_value = ([], {}, {})
    with pytest.raises(RuntimeError, match="curated gene list"):
        pipe.process_antibody_jobs(topic="ATP:0000096", jobs=[_job()])
    patched_pipeline["send"].assert_not_called()
