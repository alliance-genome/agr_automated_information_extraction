import argparse
import copy
import logging
import os
import sys
import re
from collections import Counter

import dill
import requests
from transformers import pipeline

from utils.abc_utils import load_all_jobs, get_cached_mod_abbreviation_from_id, get_tet_source_id, download_abc_model, \
    download_tei_files_for_references, set_job_started, set_job_success, send_entity_tag_to_abc, get_model_data, \
    set_job_failure
from utils.tei_utils import AllianceTEI

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}
_PIPE_CACHE = {}

STRAIN_TARGET_ENTITIES = {
    "AGRKB:101000000641073": ["N2", "OP50", "TJ375"],
    "AGRKB:101000000641132": ["EG4322", "GE24"],
    "AGRKB:101000000641112": ["N2", "EG6699", "JT734"],
    "AGRKB:101000000640598": [
        "XZ1515", "XZ1514", "QX1794", "PB306", "ECA369", "CX11271", "JT73",
        "JT513", "XZ1513", "JJ1271", "ECA36", "SP346", "RB2488", "ECA372",
        "NIC268", "RB1658", "NH646", "LKC34", "CB185", "JU1200", "RB1977",
        "ECA189", "JU258", "XZ1516", "JU367", "GH383", "CX11314", "QG556",
        "ECA191", "NIC256", "RT362", "WN2001", "MY10", "JU775", "BA819",
        "CB4932", "PB303", "JK4545", "OP50", "NIC251", "JU1242", "QG2075",
        "CB30", "GL302", "QX1791", "ECA396", "JT11398", "JU830", "JU363",
        "QX1793", "EG4725", "NIC199", "CB4856", "ECA363", "N2"
    ],
    "AGRKB:101000000641062": ["PD1074", "HT115"],
    "AGRKB:101000000641018": ["VC2428", "N2", "OP50", "VC1743"],
    "AGRKB:101000000640727": [
        "VC1263", "CB3203", "CB3257", "MT5013", "SP1713", "VC610", "CB3261",
        "MT5006", "RB983", "MT4433", "MT8886", "KJ462", "MT9958", "PR678",
        "CB936", "N2", "CU1715", "NG144", "RB1100", "NF87", "CU2945",
        "PR811", "PR691", "MT11068", "MT4434", "PR767"
    ],
    "AGRKB:101000000640813": ["N2", "CB4856", "JU1580"],
    "AGRKB:101000000639765": [
        "N2", "DA1814", "AQ866", "LX702", "LX703", "CX13079", "OH313",
        "VC125", "VC670", "RB785", "RB1680"
    ],
    "AGRKB:101000000640768": [
        "KG1180", "RB830", "TR2171", "ZX460", "OP50-1"
    ]
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

# generic alphanumeric+colon+underscore+dash tokens with ≥1 letter & ≥1 digit
GENERIC_NAME_PATTERN = re.compile(
    r'\b'                # word boundary
    r'(?=.*[A-Za-z])'    # at least one letter
    r'(?=.*\d)'          # at least one digit
    r'[A-Za-z0-9:_\-]+'  # allowed chars
    r'\b'
)


def find_best_tfidf_threshold(mod_id, topic, jobs, target_entities):
    mod_abbr = get_cached_mod_abbreviation_from_id(mod_id)
    entity_extraction_model_file_path = (f"/data/agr_document_classifier/biocuration_entity_extraction_{mod_abbr}_"
                                         f"{topic.replace(':', '_')}.dpkl")
    try:
        download_abc_model(mod_abbreviation=mod_abbr, topic=topic, output_path=entity_extraction_model_file_path,
                           task_type="biocuration_entity_extraction")
        logger.info(f"Classification model downloaded for mod: {mod_abbr}, topic: {topic}.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"Classification model not found for mod: {mod_abbr}, topic: {topic}. Skipping.")
            return None
        else:
            raise

    classification_batch_size = int(os.environ.get("CLASSIFICATION_BATCH_SIZE", 1000))
    jobs_to_process = copy.deepcopy(jobs)
    entity_extraction_model = dill.load(open(entity_extraction_model_file_path, "rb"))
    entity_extraction_model.alliance_entities_loaded = True

    best_threshold = 0.1
    best_similarity = -1
    thresholds = [i / 10.0 for i in range(1, 51)]

    while jobs_to_process:
        job_batch = jobs_to_process[:classification_batch_size]
        reference_curie_job_map = {job["reference_curie"]: job for job in job_batch}
        jobs_to_process = jobs_to_process[classification_batch_size:]
        logger.info(f"Processing a batch of {str(len(job_batch))} jobs. "
                    f"Jobs remaining to process: {str(len(jobs_to_process))}")

        os.makedirs("/data/agr_entity_extraction/to_extract", exist_ok=True)
        for file in os.listdir("/data/agr_entity_extraction/to_extract"):
            os.remove(os.path.join("/data/agr_entity_extraction/to_extract", file))
        download_tei_files_for_references(list(reference_curie_job_map.keys()),
                                          "/data/agr_entity_extraction/to_extract", mod_abbr)

        for threshold in thresholds:
            entity_extraction_model.tfidf_threshold = threshold
            total_similarity = 0

            for file in os.listdir("/data/agr_entity_extraction/to_extract"):
                curie = file.split(".")[0].replace("_", ":")
                try:
                    tei_obj = AllianceTEI()
                    tei_obj.load_from_file(f"/data/agr_entity_extraction/to_extract/{file}")
                except Exception as e:
                    logger.warning(f"Error loading TEI file for {curie}: {str(e)}. Skipping.")
                    continue

                nlp_pipeline = pipeline("ner", model=entity_extraction_model,
                                        tokenizer=entity_extraction_model.tokenizer)
                try:
                    fulltext = tei_obj.get_fulltext()
                except Exception as e:
                    logger.error(f"Error getting fulltext for {curie}: {str(e)}. Skipping.")
                    continue

                results = nlp_pipeline(fulltext)
                entities_in_fulltext = [result['word'] for result in results if result['entity'] == "ENTITY"]
                entities_to_extract = set(entity_extraction_model.entities_to_extract)
                entities_in_title = set(
                    entity_extraction_model.tokenizer.tokenize(tei_obj.get_title() or "")).intersection(
                    entities_to_extract)
                entities_in_abstract = set(
                    entity_extraction_model.tokenizer.tokenize(tei_obj.get_abstract() or "")).intersection(
                    entities_to_extract)
                all_entities = set(entities_in_fulltext).union(entities_in_title).union(entities_in_abstract)

                # Compute Jaccard similarity
                all_entities_lower = set(entity.lower() for entity in all_entities)
                target_set_lower = set(entity.lower() for entity in target_entities.get(curie, []))
                similarity = len(all_entities_lower.intersection(target_set_lower)) / len(
                    all_entities_lower.union(target_set_lower))
                total_similarity += similarity

            avg_similarity = total_similarity / len(reference_curie_job_map)
            logger.info(f"Threshold {threshold}: Average Jaccard similarity {avg_similarity}")

            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_threshold = threshold

    logger.info(f"Best TFIDF threshold: {best_threshold} with Jaccard similarity {best_similarity}")
    return best_threshold


def process_entity_extraction_jobs(
    mod_id,
    topic,
    jobs,
    test_mode: bool = False,
    test_fh=None,
    ner_batch_size: int = 16,
):  # noqa: C901
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

    # --- load metadata & model once ---
    try:
        meta = get_model_data(mod_abbreviation=mod_abbr,
                              task_type="biocuration_entity_extraction",
                              topic=topic)
        species = meta["species"]
        novel_data = meta["novel_topic_data"]

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
    ner_pipe = get_pipe(mod_abbr, topic, model)

    classification_batch_size = int(os.environ.get("CLASSIFICATION_BATCH_SIZE", 1000))
    jobs_to_process = copy.deepcopy(jobs)

    def build_entities_from_results(results, title, abstract, fulltext):
        """Combine NER hits + title/abstract tokens + regex fallback."""
        gold_up = {e.upper() for e in model.entities_to_extract}

        ents_full = {
            r["word"].upper()
            for r in results
            if r.get("entity_group") == "ENTITY" and r["word"].upper() in gold_up
        }

        t_ents, a_ents = extract_entities_from_title_abstract(model, title, abstract)
        ents_title = {e.upper() for e in t_ents}
        ents_abs = {e.upper() for e in a_ents}

        regex_hits = {
            m.upper()
            for m in GENERIC_NAME_PATTERN.findall(fulltext or "")
            if m.upper() in gold_up
        }

        all_up = ents_full | ents_title | ents_abs | regex_hits

        out = set()
        for tok in all_up:
            for part in tok.split():
                if part in gold_up:
                    out.add(part)
        return sorted(out)

    while jobs_to_process:
        job_batch = jobs_to_process[:classification_batch_size]
        jobs_to_process = jobs_to_process[classification_batch_size:]

        ref_map = {j["reference_curie"]: j for j in job_batch}
        logger.info("Processing batch of %d jobs. Remaining: %d",
                    len(job_batch), len(jobs_to_process))

        # Prep TEI directory
        out_dir = "/data/agr_entity_extraction/to_extract"
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))

        download_tei_files_for_references(list(ref_map.keys()), out_dir, mod_abbr)

        texts = []
        metas = []  # (curie, job, title, abstract, fulltext)
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

            texts.append(fulltext)
            metas.append((curie, job, title, abstract, fulltext))

        if not texts:
            logger.info("No valid TEIs in this batch.")
            continue

        # batched NER
        results_list = ner_pipe(texts, batch_size=ner_batch_size, truncation=True)

        for (curie, job, title, abstract, fulltext), results in zip(metas, results_list):
            all_entities = build_entities_from_results(results, title, abstract, fulltext)

            if test_mode:
                ents_str = " | ".join(all_entities)
                test_fh.write(f"{curie}\t{ents_str}\n")
                test_fh.flush()
            else:
                if not all_entities:
                    # send negated tag
                    send_entity_tag_to_abc(
                        reference_curie=curie,
                        species=species,
                        topic=topic,
                        negated=True,
                        tet_source_id=tet_source_id,
                        novel_data=novel_data
                    )
                else:
                    # send entities
                    for ent in all_entities:
                        if ent in model.name_to_curie_mapping:
                            ent_curie = model.name_to_curie_mapping[ent]
                        else:
                            ent_curie = model.name_to_curie_mapping[
                                model.upper_to_original_mapping[ent]
                            ]
                        send_entity_tag_to_abc(
                            reference_curie=curie,
                            species=species,
                            topic=topic,
                            entity_type=topic,
                            entity=ent_curie,
                            tet_source_id=tet_source_id,
                            novel_data=novel_data
                        )

            logger.info("%s = %s", curie, all_entities)
            set_job_started(job)
            set_job_success(job)

        logger.info("Finished processing batch of %d jobs.", len(job_batch))


def get_model(mod_abbr, topic, path):
    k = (mod_abbr, topic)
    if k not in _MODEL_CACHE:
        _MODEL_CACHE[k] = dill.load(open(path, "rb"))
    return _MODEL_CACHE[k]


def get_pipe(mod_abbr, topic, model):
    k = (mod_abbr, topic)
    if k not in _PIPE_CACHE:
        _PIPE_CACHE[k] = pipeline(
            "ner",
            model=model,
            tokenizer=model.tokenizer,
            aggregation_strategy="simple"
        )
    return _PIPE_CACHE[k]


def extract_entities_from_title_abstract(model, title, abstract):
    """
    prepare gold sets (mixed‐case and uppercase)
    gold: the set of all valid entity names in their original case
    gold_up: an uppercase copy, so we can match case-insensitively later
    """
    gold = set(model.entities_to_extract)
    gold_up = {g.upper() for g in gold}

    # tokenize title
    tokenized_title = model.tokenizer.tokenize(title or "")
    # take any tokens that exactly match an entry in gold
    entities_in_title = set(tokenized_title) & gold
    # uppercase all tokens and intersect with gold_up, adding any new matches
    entities_in_title |= {t.upper() for t in tokenized_title} & gold_up

    # regex fallback: find every generic token (letters+digits) in the raw title
    # uppercase it, and if it’s in gold_up, add it to our title entities
    for match in GENERIC_NAME_PATTERN.findall(title or ""):
        m = match.upper()
        if m in gold_up:
            entities_in_title.add(m)

    # tokenize abstract
    tokenized_abstract = model.tokenizer.tokenize(abstract or "")
    entities_in_abstract = set(tokenized_abstract) & gold
    entities_in_abstract |= {t.upper() for t in tokenized_abstract} & gold_up

    # regex fallback on abstract
    for match in GENERIC_NAME_PATTERN.findall(abstract or ""):
        m = match.upper()
        if m in gold_up:
            entities_in_abstract.add(m)

    return list(entities_in_title), list(entities_in_abstract)


def main():
    parser = argparse.ArgumentParser(description='Extract biological entities from documents')
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

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )

    mod_topic_jobs = load_all_jobs("_extraction_job", args=None)

    # ---- filtering by topic/mod ----
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

    # ---- tune threshold path ----
    if args.tune_threshold:
        for (mod_id, topic), jobs in mod_topic_jobs.items():
            TARGET_ENTITIES = STRAIN_TARGET_ENTITIES if topic == 'ATP:0000027' else GENE_TARGET_ENTITIES
            best = find_best_tfidf_threshold(mod_id, topic, jobs, TARGET_ENTITIES)
            logger.info(f"Best TF-IDF threshold for {mod_id}/{topic}: {best}")
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
            )
    finally:
        if test_fh:
            test_fh.close()

    logger.info("Finished processing all entity extraction jobs.")


if __name__ == '__main__':

    main()
