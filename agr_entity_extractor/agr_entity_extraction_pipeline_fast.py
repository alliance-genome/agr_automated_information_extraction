#!/usr/bin/env python3
"""
Speedups:
- Cache curated entities (names+CURIEs) per (MOD, entity_type) on disk & RAM.
- Pre-filter fulltext to only likely-entity sentences before NER.
- Batch NER safely; fallback if tokenizer lacks pad_token_id.
- Log timing/progress during each batch.
- Skip/quiet HuggingFace pipeline for unsupported custom models.

CLI:
    --tune-threshold          Tune TF-IDF threshold (slow)
    -t PATH / --test-output   Write "<curie>\t<entities>" to PATH instead of sending tags
    -T CURIE --topic CURIE    Filter topics (repeatable) eg -T ATP:0000027
    -m MOD   --mod MOD        Filter MODs (repeatable) eg -m WB
    --ner-batch INT           HF NER batch size (default 16)
    --no-prefilter            Disable regex/dictionary prefilter (slower)
    --log-every INT           Log progress every N papers (default 10)
"""
import argparse
import logging
import os
import re
import sys
import time
import copy
from typing import List
import dill
import requests
from transformers import pipeline
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

from utils.entity_extraction_utils import (
    prime_model_entities as prime_model_entities_shared,
    get_model,
    get_pipe,
    run_ner_batched,
    prefilter_text as prefilter_text_generic,
    build_entities_from_results as build_entities_from_results_generic,
    resolve_entity_curie,
    has_allele_like_context,
    rescue_short_alleles_from_fulltext,
    SUSPICIOUS_PREFIX_RE,
    ALLELE_NAME_PATTERN,
    GENERIC_NAME_PATTERN,
    STRAIN_NAME_PATTERN,
    ALLELE_TARGET_ENTITIES,
    GENE_TARGET_ENTITIES,
    STRAIN_TARGET_ENTITIES
)

# Silence HF info/warnings entirely
set_verbosity_error()

logger = logging.getLogger(__name__)

TOPIC2TYPE = {
    "ATP:0000027": "strain",
    "ATP:0000110": "transgenic_allele",
    "ATP:0000006": "allele",
    "ATP:0000123": "species",  # not used here but mapped for completeness
    "ATP:0000005": "gene",
}


def topic_to_entity_type(topic: str) -> str:
    return TOPIC2TYPE.get(topic, "gene")


def get_generic_name_pattern(topic: str) -> re.Pattern:
    if topic == "ATP:0000027":       # strain
        return STRAIN_NAME_PATTERN
    if topic == "ATP:0000006":       # allele
        return ALLELE_NAME_PATTERN
    return GENERIC_NAME_PATTERN      # gene, transgene, etc.


# --------------------------------------------------------------------- #
# Priming: load curated entities into the model                         #
# --------------------------------------------------------------------- #
def prime_model_entities(model, mod_abbr: str, topic: str):
    entity_type = topic_to_entity_type(topic)
    prime_model_entities_shared(
        model=model,
        mod_abbr=mod_abbr,
        entity_type=entity_type,
        loader_fn=get_all_curated_entities,
    )


# --------------------------------------------------------------------- #
# Prefilter + Build helpers (via shared utils)                          #
# --------------------------------------------------------------------- #
def prefilter_text(fulltext: str, model) -> str:
    return prefilter_text_generic(
        fulltext=fulltext,
        model=model,
        pattern=get_generic_name_pattern(model.topic),
        is_species=False,               # no species-specific normalization
        normalize_aliases=None,
        expand_abbrevs=None,
        use_gold_substring=False,
    )


def build_entities_from_results(results, title: str, abstract: str, fulltext: str, model) -> List[str]:
    entities = build_entities_from_results_generic(
        results=results,
        title=title,
        abstract=abstract,
        fulltext=fulltext,
        model=model,
        pattern=get_generic_name_pattern(model.topic),
        is_species=False,
        normalize_aliases=None,
        expand_abbrevs=None,
        use_fulltext_tokenizer=False,
        use_count_gate=False,
    )

    # For allele topic, enforce allele-specific post-processing rules.
    if getattr(model, "topic", None) == "ATP:0000006":
        original_count = len(entities)

        # 1) Keep canonical lowercase allele-like tokens or weird non-alnum mixes
        # str.isalnum() checks whether all characters in the string are alphanumeric (A–Z, a–z, 0–9)
        # and there is at least one character.
        entities = [
            e for e in entities
            if e.islower() or not e.replace('-', '').replace('_', '').isalnum()
        ]

        # 2) Drop suspicious one-letter+digit alleles (e5, e7, b2, ...) unless context looks allele-like
        before_suspicious_filter = len(entities)
        filtered_entities: list[str] = []
        for e in entities:
            if SUSPICIOUS_PREFIX_RE.match(e):
                if not has_allele_like_context(fulltext, e):
                    logger.info(
                        "ALLELE-FILTER: dropping suspicious short allele '%s' due to weak context",
                        e,
                    )
                    continue
            filtered_entities.append(e)
        entities = filtered_entities
        dropped_suspicious = before_suspicious_filter - len(entities)

        # 3) Rescue allele-like tokens from fulltext that NER missed
        already_found = set(entities)
        rescued = rescue_short_alleles_from_fulltext(fulltext, model, already_found)
        if rescued:
            logger.info(
                "ALLELE-RESCUE: adding %d alleles from fulltext that NER missed: %s",
                len(rescued), ", ".join(sorted(rescued))
            )
            entities.extend(sorted(rescued))

        # 4) Normalize and dedupe after rescue
        normalized: list[str] = []
        seen: set[str] = set()
        for e in entities:
            e_norm = e.rstrip('.')
            if e_norm and e_norm not in seen:
                seen.add(e_norm)
                normalized.append(e_norm)
        entities = normalized

        # 5) Filter to curated allele list only (case-insensitive)
        curated = getattr(model, "entities_to_extract", None)
        dropped_non_curated = 0
        if curated:
            curated_lower = {c.lower() for c in curated if isinstance(c, str)}
            before_curated_filter = len(entities)
            entities = [e for e in entities if e.lower() in curated_lower]
            dropped_non_curated = before_curated_filter - len(entities)

        # 6) Map back to canonical display names using the curated mapping
        #    so mgdf50 -> mgDf50, ttti5605 -> ttTi5605, etc.
        u2o = getattr(model, "upper_to_original_mapping", {}) or {}
        canonical: list[str] = []
        seen_disp: set[str] = set()
        for e in entities:
            key = e.upper()
            display = u2o.get(key, e)  # fall back to e if not in mapping
            if display not in seen_disp:
                seen_disp.add(display)
                canonical.append(display)
        entities = canonical

        if dropped_suspicious > 0 or dropped_non_curated > 0:
            logger.info(
                "ALLELE-FILTER: start=%d, after_lowercase=%d, "
                "dropped_suspicious=%d, dropped_non_curated=%d, final=%d",
                original_count,
                before_suspicious_filter,
                dropped_suspicious,
                dropped_non_curated,
                len(entities),
            )

    return entities


# --------------------------------------------------------------------- #
# Threshold tuning                                                      #
# --------------------------------------------------------------------- #
def find_best_tfidf_threshold(mod_id, topic, jobs, target_entities):
    mod_abbr = get_cached_mod_abbreviation_from_id(mod_id)
    model_fp = f"/data/agr_document_classifier/biocuration_entity_extraction_{mod_abbr}_{topic.replace(':', '_')}.dpkl"
    try:
        download_abc_model(mod_abbreviation=mod_abbr, topic=topic, output_path=model_fp,
                           task_type="biocuration_entity_extraction")
        logger.info("Classification model downloaded for mod=%s, topic=%s.", mod_abbr, topic)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning("Model not found for mod=%s, topic=%s. Skipping.", mod_abbr, topic)
            return None
        raise

    batch_size = int(os.environ.get("CLASSIFICATION_BATCH_SIZE", 1000))
    jobs_to_process = copy.deepcopy(jobs)
    entity_model = dill.load(open(model_fp, "rb"))
    entity_model.alliance_entities_loaded = True

    best_threshold = 0.1
    best_similarity = -1.0
    thresholds = [i / 10.0 for i in range(1, 51)]

    while jobs_to_process:
        batch = jobs_to_process[:batch_size]
        ref_map = {job["reference_curie"]: job for job in batch}
        jobs_to_process = jobs_to_process[batch_size:]
        logger.info("Processing a batch of %d jobs. Remaining: %d", len(batch), len(jobs_to_process))

        out_dir = "/data/agr_entity_extraction/to_extract"
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        download_tei_files_for_references(list(ref_map.keys()), out_dir, mod_abbr)

        for th in thresholds:
            entity_model.tfidf_threshold = th
            total_sim = 0.0
            count = 0

            for f in os.listdir(out_dir):
                curie = f.split(".")[0].replace("_", ":")
                try:
                    tei = AllianceTEI()
                    tei.load_from_file(os.path.join(out_dir, f))
                except Exception as e:
                    logger.warning("Error loading TEI for %s: %s. Skipping.", curie, e)
                    continue

                nlp = pipeline("ner", model=entity_model, tokenizer=entity_model.tokenizer)
                try:
                    if entity_model.topic == "ATP:0000006":   # allele
                        fulltext = tei.get_fulltext(include_attributes=True)
                    else:
                        fulltext = tei.get_fulltext()
                except Exception as e:
                    logger.error("Error getting fulltext for %s: %s. Skipping.", curie, e)
                    continue

                results = nlp(fulltext)
                ents_ft = [r['word'] for r in results if r.get('entity') == "ENTITY"]
                ents_to_extract = set(entity_model.entities_to_extract)
                ents_title = set(entity_model.tokenizer.tokenize(tei.get_title() or "")) & ents_to_extract
                ents_abs = set(entity_model.tokenizer.tokenize(tei.get_abstract() or "")) & ents_to_extract
                all_ents = set(ents_ft) | ents_title | ents_abs

                all_low = {e.lower() for e in all_ents}
                gold_low = {e.lower() for e in target_entities.get(curie, [])}
                if all_low or gold_low:
                    sim = len(all_low & gold_low) / len(all_low | gold_low)
                    total_sim += sim
                    count += 1

            avg_sim = (total_sim / count) if count else 0.0
            logger.info("Threshold %.1f: Average Jaccard %.4f", th, avg_sim)

            if avg_sim > best_similarity:
                best_similarity = avg_sim
                best_threshold = th

    logger.info("Best TFIDF threshold: %.1f (Jaccard %.4f)", best_threshold, best_similarity)
    return best_threshold


# --------------------------------------------------------------------- #
# Core processing                                                       #
# --------------------------------------------------------------------- #
def process_entity_extraction_jobs(mod_id, topic, jobs, test_mode: bool = False, test_fh=None, ner_batch_size: int = 16, prefilter: bool = True, log_every: int = 10, combined_tei_dir: bool = False):  # noqa: C901
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
        data_novelty = meta['data_novelty']
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

    ner_pipe = get_pipe(mod_abbr, topic, model)

    # ---------------- combined TEI dir path ---------------- #
    if combined_tei_dir:
        logger.info("Processing all combined TEI files in %s (files will not be deleted)", combined_tei_dir)
        for fname in sorted(os.listdir(combined_tei_dir)):
            if not fname.endswith(".tei"):
                continue
            curie = fname.replace(".tei", "")
            path = os.path.join(combined_tei_dir, fname)
            try:
                tei = AllianceTEI()
                tei.load_from_file(path)
                title = tei.get_title() or ""
                abstract = tei.get_abstract() or ""
                if model.topic == "ATP:0000006":   # allele
                    fulltext = tei.get_fulltext(include_attributes=True)
                else:
                    fulltext = tei.get_fulltext() or ""
            except Exception as e:
                logger.warning("Failed to load/process %s: %s", fname, e)
                continue

            text_for_ner = prefilter_text(fulltext, model) if prefilter else fulltext
            results = run_ner_batched(ner_pipe, [text_for_ner], ner_batch_size)[0]
            all_entities = build_entities_from_results(results, title, abstract, fulltext, model)

            if test_mode:
                test_fh.write(f"{curie}\t{' | '.join(all_entities)}\n")
                test_fh.flush()
            else:
                if not all_entities:
                    send_entity_tag_to_abc(
                        reference_curie=curie,
                        species=species,
                        topic=topic,
                        negated=True,
                        tet_source_id=tet_source_id,
                        data_novelty=data_novelty,
                        ml_model_id=ml_model_id
                    )
                else:
                    seen = set()
                    for ent in all_entities:
                        # Allele-specific mapping debug / behavior
                        if topic == "ATP:0000006":
                            try:
                                ent_curie = resolve_entity_curie(model, ent, strict=True)
                            except KeyError:
                                logger.debug(
                                    "ALLELE-REJECT: mapping KeyError for candidate '%s' in ref %s (combined TEI)",
                                    ent, curie,
                                )
                                continue
                            if not ent_curie:
                                logger.debug(
                                    "ALLELE-REJECT: no CURIE found for candidate '%s' in ref %s (combined TEI)",
                                    ent, curie,
                                )
                                continue
                        else:
                            try:
                                ent_curie = resolve_entity_curie(model, ent, strict=True)
                            except KeyError:
                                raise

                        if not ent_curie or ent_curie in seen:
                            continue
                        seen.add(ent_curie)
                        send_entity_tag_to_abc(
                            reference_curie=curie,
                            species=species,
                            topic=topic,
                            entity_type=topic,
                            entity=ent_curie,
                            tet_source_id=tet_source_id,
                            data_novelty=data_novelty,
                            ml_model_id=ml_model_id
                        )

            logger.info("%s => %s", curie, all_entities)
        logger.info("Finished processing combined TEI directory.")
        return

    # ---------------- job-based processing ---------------- #
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

        metas: List[tuple[str, dict, str, str, str]] = []  # (curie, job, title, abstract, fulltext)
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
                if model.topic == "ATP:0000006":   # allele
                    fulltext = tei.get_fulltext(include_attributes=True)
                else:
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

            text_for_ner = prefilter_text(fulltext, model) if prefilter else fulltext
            texts_for_ner.append(text_for_ner)
            metas.append((curie, job, title, abstract, fulltext))

        if not texts_for_ner:
            logger.info("No valid TEIs in this batch.")
            continue

        # ---- Run NER ----
        t0 = time.perf_counter()
        results_list = run_ner_batched(ner_pipe, texts_for_ner, ner_batch_size)
        total_time = time.perf_counter() - t0
        logger.info("NER on %d docs took %.1fs (%.2fs/doc)", len(texts_for_ner), total_time, total_time / len(texts_for_ner))

        # ---- Post-process ----
        for idx, ((curie, job, title, abstract, fulltext), results) in enumerate(zip(metas, results_list), 1):
            all_entities = build_entities_from_results(results, title, abstract, fulltext, model)

            if test_mode:
                test_fh.write(f"{curie}\t{' | '.join(all_entities)}\n")
                test_fh.flush()
            else:
                if not all_entities:
                    send_entity_tag_to_abc(
                        reference_curie=curie,
                        species=species,
                        topic=topic,
                        negated=True,
                        tet_source_id=tet_source_id,
                        data_novelty=data_novelty,
                        ml_model_id=ml_model_id
                    )
                else:
                    seen = set()
                    for ent in all_entities:
                        # Allele-specific mapping debug / behavior
                        if topic == "ATP:0000006":
                            try:
                                ent_curie = resolve_entity_curie(model, ent, strict=True)
                            except KeyError:
                                logger.debug(
                                    "ALLELE-REJECT: mapping KeyError for candidate '%s' in ref %s",
                                    ent, curie,
                                )
                                continue
                            if not ent_curie:
                                logger.debug(
                                    "ALLELE-REJECT: no CURIE found for candidate '%s' in ref %s",
                                    ent, curie,
                                )
                                continue
                        else:
                            try:
                                ent_curie = resolve_entity_curie(model, ent, strict=True)
                            except KeyError:
                                raise

                        if not ent_curie or ent_curie in seen:
                            continue
                        seen.add(ent_curie)
                        send_entity_tag_to_abc(
                            reference_curie=curie,
                            species=species,
                            topic=topic,
                            entity_type=topic,
                            entity=ent_curie,
                            tet_source_id=tet_source_id,
                            data_novelty=data_novelty,
                            ml_model_id=ml_model_id
                        )

            if idx % log_every == 0:
                logger.info("Processed %d/%d in this batch", idx, len(metas))

            set_job_started(job)
            set_job_success(job)
            logger.info("%s = %s", curie, all_entities)

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
    parser.add_argument("--tune-threshold", action="store_true",
                        help="Run find_best_tfidf_threshold on all jobs (slow, for experimentation only)")
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
    parser.set_defaults(strict_mapping=True)

    args = parser.parse_args()

    # ---------- NEW DEFAULTS WHEN no --mod or no -- topic provided ----------
    DEFAULT_MODS = ['WB']
    # strain, gene, transgene, allele
    DEFAULT_TOPICS = ['ATP:0000027', 'ATP:0000005', 'ATP:0000110', 'ATP:0000006']
    if not args.mod:
        args.mod = DEFAULT_MODS
        logging.getLogger(__name__).info("No --mod provided; defaulting to %s", ", ".join(DEFAULT_MODS))
    if not args.topic:
        args.topic = DEFAULT_TOPICS
        logging.getLogger(__name__).info("No --topic provided; defaulting to %s", ", ".join(DEFAULT_TOPICS))
    # ------------------------------------------------------------------------

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
        if wanted_mods and mod_id_to_abbr(mod_id) not in wanted_mods:
            continue
        filtered[(mod_id, topic)] = jobs
    mod_topic_jobs = filtered

    if not mod_topic_jobs:
        logger.warning("No jobs matched the provided filters (topic/mod). Exiting.")
        return

    if args.tune_threshold:
        for (mod_id, topic), jobs in mod_topic_jobs.items():
            if topic == 'ATP:0000027':
                TARGET = STRAIN_TARGET_ENTITIES
            elif topic == 'ATP:0000006':
                TARGET = ALLELE_TARGET_ENTITIES
            else:
                TARGET = GENE_TARGET_ENTITIES
            best = find_best_tfidf_threshold(mod_id, topic, jobs, TARGET)
            logger.info("Best TF-IDF threshold for %s/%s: %s", mod_id, topic, best)
        logger.info("Threshold tuning complete.")
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
            )
    finally:
        if test_fh:
            test_fh.close()

    logger.info("Finished processing all entity extraction jobs.")


if __name__ == '__main__':
    main()
