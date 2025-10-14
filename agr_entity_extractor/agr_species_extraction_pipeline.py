import argparse
import copy
import logging
import os
import re
import sys
import time
from typing import List, Tuple
import requests
from transformers.utils.logging import set_verbosity_error

from utils.abc_utils import (
    load_all_jobs,
    get_cached_mod_abbreviation_from_id,
    get_tet_source_id,
    download_abc_model,
    download_tei_files_for_references,
    set_job_started,
    set_job_success,
    send_entity_tag_to_abc,
    get_model_data,
    set_job_failure,
    get_all_curated_entities,
)
from utils.tei_utils import AllianceTEI
from utils.species_text_norm import (
    normalize_species_aliases,
    expand_species_abbreviations,
)
from utils.species_taxon import get_parent_children_map
from utils.entity_extraction_utils import (
    ENTITY_CACHE_DIR,
    prime_model_entities as prime_model_entities_shared,
    get_model,
    get_pipe,
    run_ner_batched,
    prefilter_text as prefilter_text_generic,
    build_entities_from_results as build_entities_from_results_generic,
    names_to_curies,
    compute_cache_key,
    build_parent_reverse_index,
    prune_to_most_specific,
    curies_to_display_names,
)

# Silence HF info/warnings entirely
set_verbosity_error()

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# CONSTANTS                                                             #
# --------------------------------------------------------------------- #

SPECIES_NAME_PATTERN = re.compile(
    r'''
    (?<![A-Za-z0-9])                # left boundary: not preceded by letter/digit
    (?:[A-Z][a-z]+|[A-Z][a-z]*[a-z]) # genus or family name (capitalized word)
    (?:                              # optionally followed by:
        \s(?:[a-z]{2,}|sp\.|[A-Z][a-z]+)   # lowercase epithet, 'sp.', or second capitalized word
        (?:\s(?:[a-z]{2,}|sp\.|[A-Z][a-z]+))*  # optional more words
    )?
    (?:\s[A-Za-z0-9-]{1,})*          # optional strain/variant/alphanumeric parts
    (?![A-Za-z0-9])                  # right boundary: not followed by letter/digit
    ''',
    re.VERBOSE | re.IGNORECASE
)
SPECIES_TOPIC = "ATP:0000123"


def prime_model_entities(model, mod_abbr: str, topic: str):
    # Species-only
    prime_model_entities_shared(
        model,
        mod_abbr,
        "species",
        loader_fn=get_all_curated_entities,
    )


def prefilter_text_species(fulltext: str, model) -> str:
    return prefilter_text_generic(
        fulltext=fulltext,
        model=model,
        pattern=SPECIES_NAME_PATTERN,
        is_species=True,
        normalize_aliases=normalize_species_aliases,
        expand_abbrevs=expand_species_abbreviations,
    )


def build_entities_from_results(results, title: str, abstract: str, fulltext: str, model) -> List[str]:
    return build_entities_from_results_generic(
        results=results,
        title=title,
        abstract=abstract,
        fulltext=fulltext,
        model=model,
        pattern=SPECIES_NAME_PATTERN,
        is_species=True,
        normalize_aliases=normalize_species_aliases,
        expand_abbrevs=expand_species_abbreviations,
    )


# ------------------------------------------------------------------------- #
# Core processing: This function drives the entity extraction workflow for  #
# biocuration, processing TEI (Text Encoding Initiative) XML files and      #
# applying an NER (Named Entity Recognition) model to extract entities      #
# for a specific mod_id and topic.                                          #
# ------------------------------------------------------------------------- #
def process_entity_extraction_jobs(mod_id, topic, jobs, test_mode: bool = False, test_fh=None, ner_batch_size: int = 16, prefilter: bool = True, log_every: int = 10, combined_tei_dir: bool = False, refresh_taxa_cache: bool = False):  # noqa: C901
    mod_abbr = get_cached_mod_abbreviation_from_id(mod_id)

    tet_source_id = None
    if not test_mode:
        tet_source_id = get_tet_source_id(
            mod_abbreviation=mod_abbr,
            source_method="abc_entity_extractor",
            source_description=("Alliance entity extraction pipeline using machine learning "
                                "to identify papers of interest for curation data types")
        )

    model_fp = f"/data/agr_document_classifier/biocuration_entity_extraction_{mod_abbr}_{topic.replace(':', '_')}.dpkl"

    try:
        meta = get_model_data(mod_abbreviation=mod_abbr,
                              task_type="biocuration_entity_extraction",
                              topic=topic)
        species = meta["species"]
        novel_data = meta["novel_topic_data"]
        novel_topic_qualifier = meta['novel_topic_qualifier']
        ml_model_id = meta['ml_model_id']
        download_abc_model(mod_abbreviation=mod_abbr,
                           topic=topic,
                           output_path=model_fp,
                           task_type="biocuration_entity_extraction")
        logger.info("Classification model downloaded for mod=%s, topic=%s", mod_abbr, topic)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning("Model not found for mod=%s, topic=%s. Skipping.", mod_abbr, topic)
            return
        raise

    model = get_model(mod_abbr, topic, model_fp)
    if not getattr(model, "alliance_entities_loaded", False):
        logger.info("Priming Alliance entity lists for %s/%s", mod_abbr, topic)
        prime_model_entities(model, mod_abbr, topic)

    # get HF token - classification pipeline
    ner_pipe = get_pipe(mod_abbr, topic, model)

    # Build parent -> children map once per run (still uses on - disk cache).
    parent_to_children = {}
    rev_index = {}
    has_parent_edges = False
    if not test_mode:
        curated_curies = model.name_to_curie_mapping.values()
        parent_children_cache_key = compute_cache_key(curated_curies)
        parent_to_children = get_parent_children_map(
            species_name_to_curie=model.name_to_curie_mapping,
            api_timeout=15.0,
            api_retries=3,
            api_sleep=0.3,
            cache_dir=str(ENTITY_CACHE_DIR),
            cache_key=parent_children_cache_key,
            force_refresh=refresh_taxa_cache,
        )
        has_parent_edges = bool(parent_to_children)
        rev_index = build_parent_reverse_index(parent_to_children) if has_parent_edges else {}
    if not has_parent_edges:
        logger.warning("Parent/child map is empty — ancestor pruning disabled for this run. Aborting.")
        return

    # ---------------- NEW combined TEI DIR handling -------------------- #
    if combined_tei_dir:
        logger.info(
            "Processing all combined TEI files in %s (files will not be deleted)",
            combined_tei_dir
        )
        for fname in sorted(os.listdir(combined_tei_dir)):
            if not fname.endswith(".combined.tei"):
                continue
            curie = fname[:-len(".combined.tei")].replace("_", ":")
            path = os.path.join(combined_tei_dir, fname)
            try:
                tei = AllianceTEI()
                tei.load_from_file(path)
                title = tei.get_title() or ""
                abstract = tei.get_abstract() or ""
                fulltext = tei.get_fulltext() or ""
            except Exception as e:
                logger.warning("Failed to load/process %s: %s", fname, e)
                continue

            text_for_ner = prefilter_text_species(fulltext, model) if prefilter else fulltext
            results = run_ner_batched(ner_pipe, [text_for_ner], ner_batch_size)[0]

            # Build final entities with min_matches gating and normalization
            all_entities = build_entities_from_results(
                results=results,
                title=title,
                abstract=abstract,
                fulltext=fulltext,
                model=model
            )

            mapped = names_to_curies(model, all_entities)
            entity_curies, dropped = (
                prune_to_most_specific(mapped, rev_index)
                if has_parent_edges else (mapped, set())
            )
            deduped_names = curies_to_display_names(model.name_to_curie_mapping, entity_curies)

            if test_mode:
                test_fh.write(f"{curie}\t{' | '.join(deduped_names)}\n")
                test_fh.flush()
            else:
                if not entity_curies:
                    send_entity_tag_to_abc(
                        reference_curie=curie,
                        species=species,
                        topic=topic,
                        negated=True,
                        tet_source_id=tet_source_id,
                        novel_data=novel_data,
                        novel_topic_qualifier=novel_topic_qualifier,
                        ml_model_id=ml_model_id
                    )
                else:
                    for ent_curie in entity_curies:
                        send_entity_tag_to_abc(
                            reference_curie=curie,
                            species=species,
                            topic=topic,
                            entity_type=topic,
                            entity=ent_curie,
                            tet_source_id=tet_source_id,
                            novel_data=novel_data,
                            novel_topic_qualifier=novel_topic_qualifier,
                            ml_model_id=ml_model_id
                        )

            if dropped:
                for d in sorted(dropped):
                    logger.info("%s: dropping ancestor '%s' (has a more specific descendant)", curie, d)
            logger.info("%s (raw)  = %s", curie, all_entities)
            logger.info("%s (dedup)= %s", curie, deduped_names)
        logger.info("Finished processing combined TEI directory.")
        return

    # --------------------------------------------------------------------- #
    # job-based processing                                                  #
    # --------------------------------------------------------------------- #

    classification_batch_size = int(os.environ.get("CLASSIFICATION_BATCH_SIZE", 1000))
    jobs_to_process = copy.deepcopy(jobs)

    while jobs_to_process:
        job_batch = jobs_to_process[:classification_batch_size]
        jobs_to_process = jobs_to_process[classification_batch_size:]

        ref_map = {j["reference_curie"]: j for j in job_batch}
        logger.info("Processing batch of %d jobs. Remaining: %d", len(job_batch), len(jobs_to_process))

        out_dir = "/data/agr_entity_extraction/to_extract"
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))

        download_tei_files_for_references(list(ref_map.keys()), out_dir, mod_abbr)

        metas: List[Tuple[str, dict, str, str, str]] = []  # (curie, job, title, abstract, fulltext)
        texts_for_ner: List[str] = []

        # ---- Prepare TEIs ----
        for fname in os.listdir(out_dir):
            curie = fname.split(".")[0].replace("_", ":")
            job = ref_map.get(curie)
            if job is None:
                continue

            try:
                tei = AllianceTEI()
                tei.load_from_file(os.path.join(out_dir, fname))
            except Exception as e:
                logger.warning("TEI load failed for %s: %s. Skipping.", curie, e)
                continue

            try:
                fulltext = tei.get_fulltext() or ""
            except Exception as e:
                logger.error("Fulltext error for %s: %s. Marking failure.", curie, e)
                set_job_started(job)
                set_job_failure(job)
                continue

            try:
                abstract = tei.get_abstract() or ""
            except Exception as e:
                logger.warning("Abstract error for %s: %s. Ignoring.", curie, e)
                abstract = ""
            try:
                title = tei.get_title() or ""
            except Exception as e:
                logger.warning("Title error for %s: %s. Ignoring.", curie, e)
                title = ""

            text_for_ner = prefilter_text_species(fulltext, model) if prefilter else fulltext

            texts_for_ner.append(text_for_ner)
            metas.append((curie, job, title, abstract, fulltext))

        if not texts_for_ner:
            logger.info("No valid TEIs in this batch.")
            continue

        # ---- Run NER ----
        t0 = time.perf_counter()
        results_list = run_ner_batched(ner_pipe, texts_for_ner, ner_batch_size)
        total_time = time.perf_counter() - t0
        logger.info("NER on %d docs took %.1fs (%.2fs/doc)",
                    len(texts_for_ner), total_time, total_time / len(texts_for_ner))

        # ---- Post-process ----
        for idx, ((curie, job, title, abstract, fulltext), results) in enumerate(zip(metas, results_list), 1):
            all_entities = build_entities_from_results(results, title, abstract, fulltext, model)

            mapped = names_to_curies(model, all_entities)
            entity_curies, dropped = (
                prune_to_most_specific(mapped, rev_index)
                if has_parent_edges else (mapped, set())
            )
            deduped_names = curies_to_display_names(model.name_to_curie_mapping, entity_curies)

            if test_mode:
                test_fh.write(f"{curie}\t{' | '.join(deduped_names)}\n")
                test_fh.flush()
            else:
                if not entity_curies:
                    send_entity_tag_to_abc(
                        reference_curie=curie,
                        species=species,
                        topic=topic,
                        negated=True,
                        tet_source_id=tet_source_id,
                        novel_data=novel_data,
                        novel_topic_qualifier=novel_topic_qualifier,
                        ml_model_id=ml_model_id
                    )
                else:
                    for ent_curie in entity_curies:
                        send_entity_tag_to_abc(
                            reference_curie=curie,
                            species=species,
                            topic=topic,
                            entity_type=topic,
                            entity=ent_curie,
                            tet_source_id=tet_source_id,
                            novel_data=novel_data,
                            novel_topic_qualifier=novel_topic_qualifier,
                            ml_model_id=ml_model_id
                        )
            if idx % log_every == 0:
                logger.info("Processed %d/%d in this batch", idx, len(metas))

            set_job_started(job)
            set_job_success(job)

            if dropped:
                for d in sorted(dropped):
                    logger.info("%s: dropping ancestor '%s' (has a more specific descendant)", curie, d)
            logger.info("%s (raw)  = %s", curie, all_entities)
            logger.info("%s (dedup)= %s", curie, deduped_names)

        logger.info("Finished processing batch of %d jobs.", len(job_batch))


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description='Extract biological entities from documents')
    parser.add_argument(
        "--combined-tei-dir", metavar="DIR",
        help="Process every .combined.tei under this directory instead of downloading TEIs."
    )
    parser.add_argument("--refresh-taxa-cache", action="store_true",
                        help="Ignore on-disk parent/children cache and rebuild from NCBI API")
    parser.add_argument("-l", "--log_level", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="Set the logging level")
    parser.add_argument("-t", "--test-output", metavar="PATH",
                        help="Write '<curie>\\t<pipe_separated_entities>' to PATH and skip sending tags to ABC")
    parser.add_argument("-T", "--topic", action="append",
                        help="Only process these topic CURIE(s). Repeatable.")
    parser.add_argument("-m", "--mod", action="append",
                        help="Only process these MOD abbreviations (e.g. WB, ZFIN). Repeatable.")
    parser.add_argument("--ner-batch", type=int, default=16,
                        help="Batch size for the HuggingFace NER pipeline (default: 16).")
    parser.add_argument("--no-prefilter", action="store_true",
                        help="Disable regex/dictionary prefiltering before NER (slower).")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log progress every N papers inside a batch (default: 10).")

    args = parser.parse_args()

    # ---------------- Default mod/topic if not provided ---------------- #
    DEFAULT_MODS = ['WB']
    DEFAULT_TOPICS = ['ATP:0000123']  # species
    if not args.mod:
        args.mod = DEFAULT_MODS
        logging.getLogger(__name__).info("No --mod provided; defaulting to %s", ', '.join(DEFAULT_MODS))
    if not args.topic:
        args.topic = DEFAULT_TOPICS
        logging.getLogger(__name__).info("No --topic provided; defaulting to %s", ', '.join(DEFAULT_TOPICS))
    # ------------------------------------------------------------------- #

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )

    mod_topic_jobs = load_all_jobs("_extraction_job", args=None)

    wanted_topics = set(args.topic) if args.topic else None
    wanted_mods = {m.upper() for m in (args.mod or [])} if args.mod else None
    _mod_cache = {}

    def mod_id_to_abbr(mod_id):
        if mod_id not in _mod_cache:
            _mod_cache[mod_id] = get_cached_mod_abbreviation_from_id(mod_id).upper()
        return _mod_cache[mod_id]

    filtered = {}
    for (mod_id, topic), jobs in mod_topic_jobs.items():
        if wanted_topics and topic not in wanted_topics:
            continue
        # Enforce species-only pipeline
        if topic != SPECIES_TOPIC:
            logger.warning("Skipping non-species topic %s (species-only pipeline).", topic)
            continue
        if wanted_mods and mod_id_to_abbr(mod_id) not in wanted_mods:
            continue
        filtered[(mod_id, topic)] = jobs
    mod_topic_jobs = filtered

    if not mod_topic_jobs:
        logger.warning("No jobs matched the provided filters (topic/mod). Exiting.")
        return

    test_mode = bool(args.test_output)
    test_fh = open(args.test_output, "w", encoding="utf-8") if test_mode else None
    try:
        for (mod_id, topic), jobs in mod_topic_jobs.items():
            process_entity_extraction_jobs(
                mod_id,
                topic,
                jobs,
                test_mode=test_mode,
                test_fh=test_fh,
                ner_batch_size=args.ner_batch,
                prefilter=not args.no_prefilter,
                log_every=args.log_every,
                combined_tei_dir=args.combined_tei_dir,
                refresh_taxa_cache=args.refresh_taxa_cache,
            )
    finally:
        if test_fh:
            test_fh.close()

    logger.info("Finished processing all entity extraction jobs.")


if __name__ == '__main__':
    main()
