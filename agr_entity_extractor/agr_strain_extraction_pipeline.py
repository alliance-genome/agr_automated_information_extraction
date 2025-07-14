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


def process_entity_extraction_jobs(mod_id, topic, jobs):  # noqa: C901
    mod_abbr = get_cached_mod_abbreviation_from_id(mod_id)
    tet_source_id = get_tet_source_id(mod_abbreviation=mod_abbr, source_method="abc_entity_extractor",
                                      source_description="Alliance entity extraction pipeline using machine learning "
                                                         "to identify papers of interest for curation data types")
    entity_extraction_model_file_path = (f"/data/agr_document_classifier/biocuration_entity_extraction_{mod_abbr}_"
                                         f"{topic.replace(':', '_')}.dpkl")
    try:
        model_metadata = get_model_data(mod_abbreviation=mod_abbr, task_type="biocuration_entity_extraction",
                                        topic=topic)
        species = model_metadata['species']
        novel_data = model_metadata['novel_topic_data']
        download_abc_model(mod_abbreviation=mod_abbr, topic=topic, output_path=entity_extraction_model_file_path,
                           task_type="biocuration_entity_extraction")
        logger.info(f"Classification model downloaded for mod: {mod_abbr}, topic: {topic}.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"Classification model not found for mod: {mod_abbr}, topic: {topic}. Skipping.")
            return
        else:
            raise
    classification_batch_size = int(os.environ.get("CLASSIFICATION_BATCH_SIZE", 1000))
    jobs_to_process = copy.deepcopy(jobs)
    entity_extraction_model = dill.load(open(entity_extraction_model_file_path, "rb"))
    while jobs_to_process:
        job_batch = jobs_to_process[:classification_batch_size]
        reference_curie_job_map = {job["reference_curie"]: job for job in job_batch}
        jobs_to_process = jobs_to_process[classification_batch_size:]
        logger.info(f"Processing a batch of {str(len(job_batch))} jobs. "
                    f"Jobs remaining to process: {str(len(jobs_to_process))}")
        os.makedirs("/data/agr_entity_extraction/to_extract", exist_ok=True)
        logger.info("Cleaning up existing files in the to_extract directory")
        for file in os.listdir("/data/agr_entity_extraction/to_extract"):
            os.remove(os.path.join("/data/agr_entity_extraction/to_extract", file))
        download_tei_files_for_references(list(reference_curie_job_map.keys()),
                                          "/data/agr_entity_extraction/to_extract", mod_abbr)
        for file in os.listdir("/data/agr_entity_extraction/to_extract"):
            curie = file.split(".")[0].replace("_", ":")
            job = reference_curie_job_map[curie]
            try:
                tei_obj = AllianceTEI()
                tei_obj.load_from_file(f"/data/agr_entity_extraction/to_extract/{file}")
            except Exception as e:
                logger.warning(f"Error loading TEI file for {curie}: {str(e)}. Skipping.")
                continue

            title = ""
            abstract = ""
            try:
                fulltext = tei_obj.get_fulltext()
            except Exception as e:
                logger.error(f"Error getting fulltext for {curie}: {str(e)}. Skipping.")
                set_job_started(job)
                set_job_failure(job)
            try:
                abstract = tei_obj.get_abstract()
            except Exception as e:
                logger.warning(f"Error getting abstract for {curie}: {str(e)}. Ignoring field.")
            try:
                title = tei_obj.get_title()
            except Exception as e:
                logger.warning(f"Error getting title for {curie}: {str(e)}. Ignoring field.")
            if topic in ['ATP:0000027', 'ATP:0000110']:
                all_entities = extract_all_entities(fulltext=fulltext,
                                                    model_path=entity_extraction_model_file_path,
                                                    title=title,
                                                    abstract=abstract)
            else:
                entity_extraction_model.load_entities_dynamically()
                nlp_pipeline = pipeline("ner", model=entity_extraction_model,
                                        tokenizer=entity_extraction_model.tokenizer)
                all_entities = extract_all_entities_with_loaded_model(
                    nlp_pipeline=nlp_pipeline,
                    fulltext=fulltext,
                    title=title,
                    abstract=abstract,
                    entity_extraction_model=entity_extraction_model
                )

            logger.info("Sending 'no data' tag to ABC.")
            if not all_entities:
                send_entity_tag_to_abc(
                    reference_curie=curie,
                    species=species,
                    topic=topic,
                    negated=True,
                    tet_source_id=tet_source_id,
                    novel_data=novel_data
                )

            logger.info("Sending extracted entities as tags to ABC.")
            for entity in all_entities:
                if entity in entity_extraction_model.name_to_curie_mapping:
                    entity_curie = entity_extraction_model.name_to_curie_mapping[entity]
                else:
                    entity_curie = entity_extraction_model.name_to_curie_mapping[
                        entity_extraction_model.upper_to_original_mapping[entity]]
                send_entity_tag_to_abc(reference_curie=curie, species=species, topic=topic, entity_type=topic,
                                       entity=entity_curie, tet_source_id=tet_source_id, novel_data=novel_data)
            logger.info(f"{curie} = {all_entities}")
            set_job_started(job)
            set_job_success(job)
        logger.info(f"Finished processing batch of {len(job_batch)} jobs.")


def extract_entities_from_title_abstract(entity_extraction_model, title, abstract):
    """
    prepare gold sets (mixed‐case and uppercase)
    gold: the set of all valid entity names in their original case
    gold_up: an uppercase copy, so we can match case-insensitively later
    """
    gold = set(entity_extraction_model.entities_to_extract)
    gold_up = {g.upper() for g in gold}

    # tokenize title
    tokenized_title = entity_extraction_model.tokenizer.tokenize(title or "")
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
    tokenized_abstract = entity_extraction_model.tokenizer.tokenize(abstract or "")
    entities_in_abstract = set(tokenized_abstract) & gold
    entities_in_abstract |= {t.upper() for t in tokenized_abstract} & gold_up

    # regex fallback on abstract
    for match in GENERIC_NAME_PATTERN.findall(abstract or ""):
        m = match.upper()
        if m in gold_up:
            entities_in_abstract.add(m)

    return list(entities_in_title), list(entities_in_abstract)


def extract_all_entities(fulltext, model_path, title, abstract):
    # load model & gold list
    entity_extraction_model = dill.load(open(model_path, "rb"))
    gold_up = {e.upper() for e in entity_extraction_model.entities_to_extract}

    # 1) NER on fulltext
    ner = pipeline(
        "ner",
        model=entity_extraction_model,
        tokenizer=entity_extraction_model.tokenizer,
        aggregation_strategy="simple"
    )
    # run NER on the full text (or empty string)
    results = ner(fulltext or "")
    # collect every recognized entity, uppercase it, and keep only those in gold_up
    entities_in_fulltext = {
        ent["word"].upper()
        for ent in results
        if ent.get("entity_group") == "ENTITY"
        and ent["word"].upper() in gold_up
    }

    # 2) title & abstract extraction
    entities_in_title, entities_in_abstract = extract_entities_from_title_abstract(
        entity_extraction_model, title, abstract
    )
    entities_in_title = {e.upper() for e in entities_in_title}
    entities_in_abstract = {e.upper() for e in entities_in_abstract}

    # 3) regex fallback on fulltext
    regex_hits = {
        match.upper()
        for match in GENERIC_NAME_PATTERN.findall(fulltext or "")
        if match.upper() in gold_up
    }

    # combine all
    all_entities = entities_in_fulltext | entities_in_title | entities_in_abstract | regex_hits

    # split multi‐token strings and dedupe
    strains = set()
    for token in all_entities:
        for part in token.split():
            if part in gold_up:
                strains.add(part)

    return sorted(strains)


def extract_all_entities_with_loaded_model(nlp_pipeline, fulltext, entity_extraction_model, title, abstract):
    results = nlp_pipeline(fulltext)
    entities_in_fulltext = [result['word'] for result in results if result['entity'] == "ENTITY"]
    entities_in_title, entities_in_abstract = extract_entities_from_title_abstract(entity_extraction_model, title, abstract)
    all_entities = set(entities_in_fulltext + entities_in_title + entities_in_abstract)
    upper_counter = Counter([entity.upper() for entity in all_entities])
    for upper_entity, count in upper_counter.items():
        if count > 1:
            all_entities.remove(upper_entity)
    return all_entities


def main():
    parser = argparse.ArgumentParser(description='Extract biological entities from documents')
    parser.add_argument("--tune-threshold", action="store_true",
                        help="Run find_best_tfidf_threshold on all jobs (slow, for experimentation only)")
    parser.add_argument("-l", "--log_level", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="Set the logging level")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )
    mod_topic_jobs = load_all_jobs("_extraction_job", args=None)
    if args.tune_threshold:
        for (mod_id, topic), jobs in mod_topic_jobs.items():
            TARGET_ENTITIES = (
                STRAIN_TARGET_ENTITIES
                if topic == 'ATP:0000027'
                else GENE_TARGET_ENTITIES
            )
            best = find_best_tfidf_threshold(mod_id, topic, jobs, TARGET_ENTITIES)
            logger.info(f"Best TF-IDF threshold for {mod_id}/{topic}: {best}")
        logger.info("Threshold tuning complete.")
        return

    for (mod_id, topic), jobs in mod_topic_jobs.items():
        process_entity_extraction_jobs(mod_id, topic, jobs)
    logger.info("Finished processing all entity extraction jobs.")


if __name__ == '__main__':

    main()
