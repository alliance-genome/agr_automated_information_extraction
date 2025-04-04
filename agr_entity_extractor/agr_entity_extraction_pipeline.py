import argparse
import copy
import logging
import os
import sys

import dill
import requests
from transformers import pipeline

from utils.abc_utils import load_all_jobs, get_cached_mod_abbreviation_from_id, get_tet_source_id, download_abc_model, \
    download_tei_files_for_references, set_job_started, set_job_success, send_entity_tag_to_abc
from utils.tei_utils import AllianceTEI

logger = logging.getLogger(__name__)


def process_entity_extraction_jobs(mod_id, topic, jobs):
    mod_abbr = get_cached_mod_abbreviation_from_id(mod_id)
    tet_source_id = get_tet_source_id(mod_abbreviation=mod_abbr, source_method="abc_entity_extractor",
                                      source_description="Alliance entity extraction pipeline using machine learning "
                                                         "to identify papers of interest for curation data types")
    entity_extraction_model_file_path = (f"/data/agr_document_classifier/biocuration_entity_extraction_{mod_abbr}_"
                                         f"{topic.replace(':', '_')}.dpkl")
    try:
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
        logger.info(f"Processing a batch of {str(classification_batch_size)} jobs. "
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
            entity_extraction_model.load_entities_dynamically_fnc()
            entity_extraction_model.alliance_entities_loaded = True
            nlp_pipeline = pipeline("ner", model=entity_extraction_model,
                                    tokenizer=entity_extraction_model.tokenizer)
            title = ""
            abstract = ""
            try:
                fulltext = tei_obj.get_fulltext()
            except Exception as e:
                logger.error(f"Error getting fulltext for {curie}: {str(e)}. Skipping.")
                continue
            try:
                abstract = tei_obj.get_abstract()
            except Exception as e:
                logger.warning(f"Error getting abstract for {curie}: {str(e)}. Ignoring field.")
            try:
                title = tei_obj.get_title()
            except Exception as e:
                logger.warning(f"Error getting title for {curie}: {str(e)}. Ignoring field.")
            results = nlp_pipeline(fulltext)
            entities_in_fulltext = [result['word'] for result in results if result['entity'] == "ENTITY"]
            entities_to_extract = set(entity_extraction_model.entities_to_extract)
            entities_to_extract_uppercase = set([entity.upper() for entity in entities_to_extract])
            tokenized_title = entity_extraction_model.tokenizer.tokenize(title)
            entities_in_title = []
            entities_in_title.extend(set(tokenized_title) & entities_to_extract)
            entities_in_title.extend(set([token.upper() for token in tokenized_title]) & entities_to_extract_uppercase)
            tokenized_abstract = entity_extraction_model.tokenizer.tokenize(abstract)
            entities_in_abstract = []
            entities_in_abstract.extend(set(tokenized_abstract) & entities_to_extract)
            entities_in_abstract.extend(set([token.upper() for token in tokenized_abstract]) & entities_to_extract_uppercase)
            all_entities = set(entities_in_fulltext + entities_in_title + entities_in_abstract)
            logger.info("Sending extracted entities as tags to ABC.")
            for entity in all_entities:
                send_entity_tag_to_abc(reference_curie=curie, mod_abbreviation=mod_abbr, topic=topic,
                                       entity=entity_extraction_model.name_to_curie_mapping[entity],
                                       tet_source_id=tet_source_id)
            set_job_started(job)
            set_job_success(job)
        logger.info(f"Finished processing batch of {len(job_batch)} jobs.")


def main():
    parser = argparse.ArgumentParser(description='Extract biological entities from documents')
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
    mod_topic_jobs = load_all_jobs("_extraction_job")
    for (mod_id, topic), jobs in mod_topic_jobs.items():
        process_entity_extraction_jobs(mod_id, topic, jobs)
    logger.info("Finished processing all entity extraction jobs.")


if __name__ == '__main__':
    main()
