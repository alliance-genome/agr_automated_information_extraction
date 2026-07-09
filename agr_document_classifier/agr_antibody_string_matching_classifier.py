"""WB antibody string-matching topic classifier (SCRUM-5601).

Pulls jobs from ABC, downloads main MD (with TEI fallback), applies the
rule engine in `antibody_rules.py`, and POSTs TopicEntityTags with
positive/negative + matched-span notes back to ABC.

Mirrors the structure of agr_document_classifier_classify.py without the
ML-model download/predict path.
"""

import argparse
import copy
import logging
import os
import os.path
import sys
import traceback
from argparse import Namespace

from agr_document_classifier.antibody_rules import build_regexes, match_antibody_spans
from utils.abc_utils import (
    download_md_files_for_references,
    get_cached_mod_id_from_abbreviation,
    get_tet_source_id,
    load_all_jobs,
    send_classification_tag_to_abc,
    set_blue_api_base_url,
    set_job_failure,
    set_job_started,
    set_job_success,
)
from utils.ateam_utils import get_all_curated_entities
from utils.md_utils import AllianceMarkdown

from agr_literature_service.lit_processing.utils.report_utils import send_report
from utils.slack_utils import send_slack_notification


logger = logging.getLogger(__name__)


# Constants per design spec §3
ANTIBODY_TOPIC = "ATP:0000096"
SOURCE_METHOD = "abc_string_matching_antibody"
SOURCE_DESCRIPTION = (
    "Alliance pipeline that identifies relevant words and/or phrases in "
    "C. elegans references to identify references describing production "
    "and/or use of antibodies."
)
JOB_LABEL = "antibody_string_matching_job"
WB_SPECIES = "NCBITaxon:6239"
DATA_NOVELTY = "ATP:0000335"  # parent term, per design spec D2

DATA_DIR = "/data/agr_document_classifier"
TO_CLASSIFY_DIR = f"{DATA_DIR}/to_classify_antibody"
STOP_FILE = f"{DATA_DIR}/stop_antibody_classifier"


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
        force=True,
    )


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(description="WB antibody string-matching topic classifier.")
    parser.add_argument("-l", "--log_level", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO')
    parser.add_argument("-f", "--reference_curie", type=str, required=False,
                        help="Run only for these comma-separated reference curies.")
    parser.add_argument("-m", "--mod_abbreviation", type=str, required=False,
                        help="Run only for this MOD (e.g. WB).")
    parser.add_argument("-t", "--topic", type=str, required=False,
                        help="Run only for this topic ATP id.")
    parser.add_argument("-s", "--stage", action="store_true",
                        help="Hit the stage ABC API.")
    parser.add_argument("--test_mode", action="store_true",
                        help="Run on the supplied references without writing TETs or "
                             "transitioning workflow status.")
    return parser.parse_args()


def _prepare_dir() -> None:
    os.makedirs(TO_CLASSIFY_DIR, exist_ok=True)
    for f in os.listdir(TO_CLASSIFY_DIR):
        try:
            os.remove(os.path.join(TO_CLASSIFY_DIR, f))
        except FileNotFoundError:
            pass


def _curie_from_filename(file_path: str) -> str:
    """`AGRKB_101000000000001.md` -> `AGRKB:101000000000001`."""
    base = os.path.basename(file_path)
    name = base.split(".")[0]
    return name.replace("_", ":")


def process_antibody_jobs(topic: str, jobs: list, *, test_mode: bool = False) -> None:
    """Process all jobs for a single topic. Used by both classify_mode
    (cron) and direct_classify_mode (--test_mode).

    Cron mode is WB-only by virtue of the `antibody_string_matching_job`
    workflow tag being wired up only for WB; test mode validates the
    --mod_abbreviation up front in direct_classify_mode.
    """
    curated_genes, _, _ = get_all_curated_entities(
        mod_abbreviation="WB", entity_type_str="gene")
    if not curated_genes:
        msg = "Empty curated gene list returned from get_all_curated_entities — aborting"
        logger.error(msg)
        raise RuntimeError(msg)

    rules = build_regexes(curated_gene_names=curated_genes)

    tet_source_id = get_tet_source_id(
        mod_abbreviation="WB",
        source_method=SOURCE_METHOD,
        source_description=SOURCE_DESCRIPTION,
    )

    batch_size = int(os.environ.get("CLASSIFICATION_BATCH_SIZE", 1000))
    pending = copy.deepcopy(jobs)
    while pending:
        if os.path.isfile(STOP_FILE):
            logger.info("stop file present — exiting between batches")
            return
        batch = pending[:batch_size]
        pending = pending[batch_size:]
        _process_batch(batch, rules, tet_source_id, topic, test_mode=test_mode)


def _process_batch(batch: list, rules, tet_source_id: int, topic: str,
                   *, test_mode: bool) -> None:
    curie_to_job = {job["reference_curie"]: job for job in batch}
    _prepare_dir()
    download_md_files_for_references(list(curie_to_job.keys()), TO_CLASSIFY_DIR, "WB")

    for fname in os.listdir(TO_CLASSIFY_DIR):
        path = os.path.join(TO_CLASSIFY_DIR, fname)
        curie = _curie_from_filename(path)
        job = curie_to_job.get(curie)
        if job is None:
            logger.warning(f"MD {fname} has no matching job; skipping")
            continue

        try:
            md = AllianceMarkdown()
            md.load_from_file(path)
            sentences = md.get_sentences()
        except Exception as exc:
            logger.error(f"MD parse failed for {curie}: {exc}")
            if not test_mode:
                set_job_started(job)
                set_job_failure(job)
            continue

        matches = match_antibody_spans(sentences, rules)
        note = ", ".join(sorted(matches)) if matches else None
        negated = len(matches) == 0

        if test_mode:
            logger.info(
                f"[test_mode] {curie}: matches={sorted(matches)} "
                f"negated={negated} note={note}"
            )
            continue

        set_job_started(job)
        ok = send_classification_tag_to_abc(
            reference_curie=curie,
            species=WB_SPECIES,
            topic=topic,
            negated=negated,
            data_novelty=DATA_NOVELTY,
            confidence_score=None,
            confidence_level=None,
            tet_source_id=tet_source_id,
            ml_model_id=None,
            note=note,
        )
        if ok:
            set_job_success(job)
        else:
            set_job_failure(job)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def classify_mode(args: Namespace) -> None:
    logger.info("Antibody string-matching classification started.")
    mod_topic_jobs = load_all_jobs(JOB_LABEL, args)
    failed = []
    for (mod_id, topic), jobs in mod_topic_jobs.items():
        try:
            process_antibody_jobs(topic, jobs)
        except Exception as exc:
            logger.error(f"Error processing antibody batch (mod={mod_id} topic={topic}): {exc}")
            failed.append({
                "mod_id": mod_id,
                "topic": topic,
                "exception": str(exc),
                "trace": "<br>".join(traceback.format_tb(exc.__traceback__)),
            })

    if failed:
        msg = "<h>Antibody string-matching pipeline — failed batches:</h><br><br>\n"
        for fp in failed:
            msg += (f"mod_id: {fp['mod_id']}, topic: {fp['topic']}<br>"
                    f"Exception: {fp['exception']}<br>"
                    f"Stacktrace: {fp['trace']}<br><br>\n")
        send_report("Antibody string-matching classification failures", msg)
        send_slack_notification("Antibody string-matching classification failures", msg)
        sys.exit(-1)


def direct_classify_mode(args: Namespace) -> None:
    if not args.reference_curie or not args.mod_abbreviation:
        logger.error("--test_mode requires --reference_curie and --mod_abbreviation")
        sys.exit(2)
    if args.mod_abbreviation != "WB":
        logger.error(f"--test_mode only supports WB; got {args.mod_abbreviation}")
        sys.exit(2)

    curies = [c.strip() for c in args.reference_curie.split(",") if c.strip()]
    mod_id = get_cached_mod_id_from_abbreviation(args.mod_abbreviation)
    jobs = [{"reference_curie": c, "reference_workflow_tag_id": None,
             "mod_id": mod_id} for c in curies]
    process_antibody_jobs(
        topic=args.topic or ANTIBODY_TOPIC,
        jobs=jobs,
        test_mode=True,
    )


def main() -> None:
    args = parse_arguments()
    configure_logging(args.log_level)
    if args.stage:
        set_blue_api_base_url("https://stage-literature-rest.alliancegenome.org")
        os.environ['ABC_API_SERVER'] = "https://stage-literature-rest.alliancegenome.org"
        os.environ["ON_PRODUCTION"] = "no"
    else:
        os.environ.setdefault("ON_PRODUCTION", "yes")

    if args.test_mode:
        direct_classify_mode(args)
    else:
        classify_mode(args)


if __name__ == "__main__":
    main()
