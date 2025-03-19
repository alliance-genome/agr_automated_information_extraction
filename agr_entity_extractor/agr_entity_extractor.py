import argparse
import copy
import logging
import os
import sys

import dill
import requests

from utils.abc_utils import load_all_jobs, get_cached_mod_abbreviation_from_id, get_tet_source_id, download_abc_model, \
    download_tei_files_for_references, set_job_started, set_job_success, get_all_curated_entities, \
    send_entity_tag_to_abc
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
    entity_type_str = ""
    if topic == "ATP:0000005":
        entity_type_str = "gene"
    curated_entities, entity_name_curie_mapping = get_all_curated_entities(mod_abbreviation=mod_abbr,
                                                                           entity_type_str=entity_type_str)
    entity_extraction_model.set_entity_list(curated_entities)
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
            tei_obj = AllianceTEI()
            tei_obj.load_from_file(f"/data/agr_entity_extraction/to_extract/{file}")

            extracted_entities = entity_extraction_model.transform(tei_obj.get_fulltext())
            logger.info("Sending extracted entities as tags to ABC.")
            for entity in extracted_entities:
                send_entity_tag_to_abc(reference_curie=curie, mod_abbreviation=mod_abbr, topic=topic,
                                       entity=entity_name_curie_mapping[entity], tet_source_id=tet_source_id)
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
