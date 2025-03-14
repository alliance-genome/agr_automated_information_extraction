import argparse
import copy
import logging
import os
import sys

import requests

from agr_document_classifier.agr_document_classifier import load_abc_model
from utils.abc_utils import load_all_jobs, get_cached_mod_abbreviation_from_id, get_tet_source_id

logger = logging.getLogger(__name__)


def process_entity_extraction_jobs(mod_id, topic, jobs):
    mod_abbr = get_cached_mod_abbreviation_from_id(mod_id)
    tet_source_id = get_tet_source_id(mod_abbreviation=mod_abbr, pipeline_name="abc_entity_extractor")
    entity_extraction_model_file_path = (f"/data/agr_document_classifier/biocuration_entity_extraction_{mod_abbr}_"
                                         f"{topic.replace(':', '_')}.joblib")
    try:
        load_abc_model(mod_abbreviation=mod_abbr, topic=topic, file_path=entity_extraction_model_file_path,
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
    classifier_model = joblib.load(classifier_file_path)
    while jobs_to_process:
        job_batch = jobs_to_process[:classification_batch_size]
        jobs_to_process = jobs_to_process[classification_batch_size:]
        logger.info(f"Processing a batch of {str(classification_batch_size)} jobs. "
                    f"Jobs remaining to process: {str(len(jobs_to_process))}")
        process_job_batch(job_batch, mod_abbr, topic, tet_source_id, embedding_model, classifier_model)


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
    mod_topic_jobs = load_all_jobs("entity_extraction_job")
    for (mod_id, topic), jobs in mod_topic_jobs.items():
        process_entity_extraction_jobs(mod_id, topic, jobs)


if __name__ == '__main__':
    main()
