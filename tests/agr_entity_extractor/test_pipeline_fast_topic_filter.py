"""Regression tests for the job/topic filter in agr_entity_extraction_pipeline_fast.main().

ABC keys WB allele extraction jobs on the classical allele topic (ATP:0000285),
while the extraction model is registered under the generic allele topic
(ATP:0000006). main() must normalize allele topics so that:
- ATP:0000285 jobs pass the default topic filter instead of being dropped;
- the model is fetched under ATP:0000006, the topic it is registered with;
- -T ATP:0000285 and -T ATP:0000006 both select allele jobs.
Without this, WB allele jobs are silently discarded before processing and no
missing-model alert is ever raised (SCRUM-6596).
"""

import sys
from unittest.mock import patch

from agr_entity_extractor import agr_entity_extraction_pipeline_fast as pipeline_fast

WB_MOD_ID = 2
ZFIN_MOD_ID = 3
MOD_ABBREVIATIONS = {WB_MOD_ID: "WB", ZFIN_MOD_ID: "ZFIN"}


def run_main(jobs_by_mod_topic, argv=None):
    """Run main() with mocked ABC access; return the (mod_id, topic, jobs) calls
    made to process_entity_extraction_jobs."""
    calls = []

    def fake_process(mod_id, topic, jobs, **kwargs):
        calls.append((mod_id, topic, jobs))
        return {"failed": 0, "md_skipped": 0, "skipped": []}

    with patch.object(pipeline_fast, "load_all_jobs", return_value=jobs_by_mod_topic), \
         patch.object(pipeline_fast, "get_cached_mod_abbreviation_from_id",
                      side_effect=MOD_ABBREVIATIONS.__getitem__), \
         patch.object(pipeline_fast, "process_entity_extraction_jobs", side_effect=fake_process), \
         patch.object(pipeline_fast, "send_slack_notification"), \
         patch.object(sys, "argv", ["agr_entity_extraction_pipeline_fast.py"] + (argv or [])):
        pipeline_fast.main()
    return calls


def test_classical_allele_jobs_are_processed_with_default_topics():
    jobs = [{"reference_curie": "AGRKB:101"}, {"reference_curie": "AGRKB:102"}]
    calls = run_main({(WB_MOD_ID, "ATP:0000285"): jobs})
    assert calls == [(WB_MOD_ID, "ATP:0000006", jobs)]


def test_generic_and_classical_allele_job_groups_are_merged():
    generic_jobs = [{"reference_curie": "AGRKB:101"}]
    classical_jobs = [{"reference_curie": "AGRKB:102"}]
    calls = run_main({
        (WB_MOD_ID, "ATP:0000006"): generic_jobs,
        (WB_MOD_ID, "ATP:0000285"): classical_jobs,
    })
    assert len(calls) == 1
    mod_id, topic, jobs = calls[0]
    assert (mod_id, topic) == (WB_MOD_ID, "ATP:0000006")
    assert sorted(j["reference_curie"] for j in jobs) == ["AGRKB:101", "AGRKB:102"]


def test_topic_flag_accepts_classical_allele_curie():
    allele_jobs = [{"reference_curie": "AGRKB:101"}]
    gene_jobs = [{"reference_curie": "AGRKB:102"}]
    calls = run_main(
        {
            (WB_MOD_ID, "ATP:0000285"): allele_jobs,
            (WB_MOD_ID, "ATP:0000005"): gene_jobs,
        },
        argv=["-T", "ATP:0000285"],
    )
    assert calls == [(WB_MOD_ID, "ATP:0000006", allele_jobs)]


def test_non_allele_topics_and_mod_filter_are_unchanged():
    gene_jobs = [{"reference_curie": "AGRKB:102"}]
    calls = run_main({
        (WB_MOD_ID, "ATP:0000005"): gene_jobs,
        (WB_MOD_ID, "ATP:0000123"): [{"reference_curie": "AGRKB:103"}],  # species: not in defaults
        (ZFIN_MOD_ID, "ATP:0000005"): [{"reference_curie": "AGRKB:104"}],  # not in default mods
    })
    assert calls == [(WB_MOD_ID, "ATP:0000005", gene_jobs)]
