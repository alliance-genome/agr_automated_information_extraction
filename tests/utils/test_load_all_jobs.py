"""Tests for load_all_jobs: grouping jobs and logging the ones it skips."""
import logging

from utils import abc_utils
from utils.abc_utils import load_all_jobs


def _job(job_name, reference_id, topic_id, mod_id):
    return {
        "job_name": job_name,
        "workflow_tag_id": "ATP:0000000",
        "reference_id": reference_id,
        "reference_curie": f"AGRKB:{reference_id}",
        "reference_workflow_tag_id": reference_id * 10,
        "mod_id": mod_id,
        "topic_id": topic_id,
    }


def _serve(monkeypatch, jobs):
    """Make get_jobs_batch page through `jobs` the way the ABC endpoint does."""
    def fake_get_jobs_batch(job_label, limit, offset, args):
        return jobs[offset:offset + limit]
    monkeypatch.setattr(abc_utils, "get_jobs_batch", fake_get_jobs_batch)


def test_groups_jobs_by_mod_and_topic_without_duplicates(monkeypatch):
    gene_1 = _job("gene_extraction_job", 1, "ATP:0000005", 1)
    gene_2 = _job("gene_extraction_job", 2, "ATP:0000005", 1)
    allele_1 = _job("allele_extraction_job", 1, "ATP:0000006", 1)
    _serve(monkeypatch, [gene_1, gene_2, dict(gene_1), allele_1])

    jobs = load_all_jobs("_extraction_job", args=None)

    assert jobs == {(1, "ATP:0000005"): [gene_1, gene_2], (1, "ATP:0000006"): [allele_1]}


def test_skipped_jobs_are_summarised_in_one_line(monkeypatch, caplog):
    # "_extraction_job" also matches email_extraction_job (SCRUM-6845), which has no topic;
    # prod had ~4,400-7,000 of them per run, each logged as its own WARNING
    gene = _job("gene_extraction_job", 1, "ATP:0000005", 1)
    emails = [_job("email_extraction_job", ref_id, None, 2) for ref_id in range(100, 2600)]
    no_mod = _job("gene_extraction_job", 3, "ATP:0000005", None)
    _serve(monkeypatch, [gene] + emails + [no_mod])

    with caplog.at_level(logging.INFO, logger=abc_utils.logger.name):
        jobs = load_all_jobs("_extraction_job", args=None)

    assert jobs == {(1, "ATP:0000005"): [gene]}
    skip_lines = [r for r in caplog.records if "Skipped" in r.getMessage()]
    assert len(skip_lines) == 1
    assert skip_lines[0].getMessage() == (
        "Skipped 2501 jobs without topic_id or mod_id: "
        "email_extraction_job=2500, gene_extraction_job=1")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_no_skip_line_when_nothing_is_skipped(monkeypatch, caplog):
    _serve(monkeypatch, [_job("gene_extraction_job", 1, "ATP:0000005", 1)])

    with caplog.at_level(logging.INFO, logger=abc_utils.logger.name):
        load_all_jobs("_extraction_job", args=None)

    assert not [r for r in caplog.records if "Skipped" in r.getMessage()]
