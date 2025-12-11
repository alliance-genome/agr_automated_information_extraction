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
    set_job_failure
)
from utils.ateam_utils import get_all_curated_entities
from utils.tei_utils import AllianceTEI

from utils.entity_extraction_utils import (
    prime_model_entities as prime_model_entities_shared,
    get_model,
    get_pipe,
    run_ner_batched,
    prefilter_text as prefilter_text_generic,
    build_entities_from_results as build_entities_from_results_generic,
    resolve_entity_curie,
)

# Silence HF info/warnings entirely
set_verbosity_error()

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# PATTERNS & TOPIC MAPPING                                              #
# --------------------------------------------------------------------- #
STRAIN_NAME_PATTERN = re.compile(
    r'''
    (?<![A-Za-z0-9])              # left-boundary: not preceded by a letter or digit
    (?=[A-Za-z0-9()_-]*[A-Za-z])  # must contain at least one letter
    (?=[A-Za-z0-9()_-]*\d)        # must contain at least one digit
    [A-Za-z0-9]                   # first char: letter or digit
    (?:                           # then any of…
       [A-Za-z0-9_-]              #   letter/digit/underscore/hyphen
     | \([A-Za-z0-9]+\)           #   "(…)" group of alphanumerics
    )*                            # repeated
    (?![A-Za-z0-9])               # right-boundary: next char is not a letter or digit
    ''',
    re.VERBOSE
)

GENERIC_NAME_PATTERN = re.compile(
    r'(?<![A-Za-z0-9_.\-])'          # left-delimiter: not letter/digit/._-
    r'(?=[A-Za-z0-9_.\-]*[A-Za-z])'  # must contain ≥1 letter
    r'(?=[A-Za-z0-9_.\-]*\d)'        # must contain ≥1 digit
    r'[A-Za-z0-9_.\-]{2,}'           # the token itself (no colon!)
    r'(?![A-Za-z0-9_.\-])'           # right-delimiter: not letter/digit/._-
)

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
    return STRAIN_NAME_PATTERN if topic == "ATP:0000027" else GENERIC_NAME_PATTERN


# --------------------------------------------------------------------- #
# Threshold-tuning gold sets (restored)                                 #
# --------------------------------------------------------------------- #
STRAIN_TARGET_ENTITIES = {
    "AGRKB:101000000641073": ["N2", "OP50", "TJ375"],
    "AGRKB:101000000641132": ["EG4322", "GE24"],
    "AGRKB:101000000641112": ["N2", "EG6699", "JT734"],
    "AGRKB:101000000640598": [
        "XZ1515", "XZ1514", "QX1794", "PB306", "ECA369", "CX11271", "JT73", "JT513", "XZ1513",
        "JJ1271", "ECA36", "SP346", "RB2488", "ECA372", "NIC268", "RB1658", "NH646", "LKC34",
        "CB185", "JU1200", "RB1977", "ECA189", "JU258", "XZ1516", "JU367", "GH383", "CX11314",
        "QG556", "ECA191", "NIC256", "RT362", "WN2001", "MY10", "JU775", "BA819", "CB4932",
        "PB303", "JK4545", "OP50", "NIC251", "JU1242", "QG2075", "CB30", "GL302", "QX1791",
        "ECA396", "JT11398", "JU830", "JU363", "QX1793", "EG4725", "NIC199", "CB4856",
        "ECA363", "N2"
    ],
    "AGRKB:101000000641062": ["PD1074", "HT115"],
    "AGRKB:101000000641018": ["VC2428", "N2", "OP50", "VC1743"],
    "AGRKB:101000000640727": [
        "VC1263", "CB3203", "CB3257", "MT5013", "SP1713", "VC610", "CB3261", "MT5006",
        "RB983", "MT4433", "MT8886", "KJ462", "MT9958", "PR678", "CB936", "N2", "CU1715",
        "NG144", "RB1100", "NF87", "CU2945", "PR811", "PR691", "MT11068", "MT4434", "PR767"
    ],
    "AGRKB:101000000640813": ["N2", "CB4856", "JU1580"],
    "AGRKB:101000000639765": [
        "N2", "DA1814", "AQ866", "LX702", "LX703", "CX13079", "OH313", "VC125", "VC670",
        "RB785", "RB1680"
    ],
    "AGRKB:101000000640768": ["KG1180", "RB830", "TR2171", "ZX460", "OP50-1"]
}

GENE_TARGET_ENTITIES = {
    "AGRKB:101000000634691": ["lov-3"],
    "AGRKB:101000000635145": ["his-58"],
    "AGRKB:101000000635933": ["spe-4", "spe-6", "spe-8", "swm-1", "zipt-7.1", "zipt-7.2"],
    "AGRKB:101000000635973": ["dot-1.1", "hbl-1", "let-7", "lin-29A", "lin-41", "mir-48", "mir-84", "mir-241"],
    "AGRKB:101000000636039": ["fog-3"],
    "AGRKB:101000000636419": [],
    "AGRKB:101000000637655": ["ddi-1", "hsf-1", "pas-1", "pbs-2", "pbs-3", "pbs-4", "pbs-5", "png-1", "rpt-3", "rpt-6",
                              "rpn-1", "rpn-5", "rpn-8", "rpn-9", "rpn-10", "rpn-11", "sel-1", "skn-1", "unc-54"],
    "AGRKB:101000000637658": ["dbt-1"],
    "AGRKB:101000000637693": ["C17D12.7", "plk-1", "spd-2", "spd-5"],
    "AGRKB:101000000637713": ["cha-1", "daf-2", "daf-16"],
    "AGRKB:101000000637764": ["F28C6.8", "Y71F9B.2", "acl-3", "crls-1", "drp-1", "fzo-1", "pgs-1"],
    "AGRKB:101000000637890": ["atg-13", "atg-18", "atg-2", "atg-3", "atg-4.1", "atg-4.2", "atg-7", "atg-9", "atg-18",
                              "asp-10", "bec-1", "ced-13", "cep-1", "egl-1", "epg-2", "epg-5", "epg-8", "epg-9",
                              "epg-11", "glh-1", "lgg-1", "pgl-1", "pgl-3", "rrf-1", "sepa-1", "vet-2", "vet-6",
                              "vha-5", "vha-13", "ZK1053.3"],
    "AGRKB:101000000637968": ["avr-15", "bet-2", "csr-1", "cye-1", "daf-12", "drh-3", "ego-1", "hrde-1", "lin-13",
                              "ncl-1", "rrf-1", "snu-23", "unc-31"],
    "AGRKB:101000000638021": [],
    "AGRKB:101000000638052": ["cept-1", "cept-2", "daf-22", "drp-1", "fat-1", "fat-2", "fat-3", "fat-4", "fat-6",
                              "fat-7", "fzo-1", "pcyt-1", "seip-1"],
}


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
    return build_entities_from_results_generic(
        results=results,
        title=title,
        abstract=abstract,
        fulltext=fulltext,
        model=model,
        pattern=get_generic_name_pattern(model.topic),
        is_species=False,               # no species-specific normalization
        normalize_aliases=None,
        expand_abbrevs=None,
        use_fulltext_tokenizer=False,
        use_count_gate=False,
    )


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
            if not fname.endswith(".combined.tei"):
                continue
            curie = fname.replace(".combined.tei", "")
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
                        help="Write '<curie>\t<pipe_separated_entities>' to PATH and skip sending tags to ABC")
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
    DEFAULT_TOPICS = ['ATP:0000027', 'ATP:0000005', 'ATP:0000110']  # strain, gene, transgene
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
            TARGET = STRAIN_TARGET_ENTITIES if topic == 'ATP:0000027' else GENE_TARGET_ENTITIES
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
