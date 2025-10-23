import argparse
import copy
import logging
import os
import os.path
import sys

import traceback
from argparse import Namespace

import joblib
import numpy as np
import requests.exceptions
from gensim.models import KeyedVectors

from utils.abc_utils import download_tei_files_for_references, send_classification_tag_to_abc, \
    get_cached_mod_abbreviation_from_id, \
    set_job_success, get_tet_source_id, set_job_started, \
    download_abc_model, set_job_failure, load_all_jobs, get_model_data, \
    get_cached_mod_species_map, set_blue_api_base_url,\
    get_cached_mod_id_from_abbreviation, send_manual_indexing_to_abc
from utils.get_documents import get_documents, remove_stopwords
from utils.embedding import load_embedding_model, get_document_embedding

from agr_literature_service.lit_processing.utils.report_utils import send_report


logger = logging.getLogger(__name__)


def configure_logging(log_level):
    # Configure logging based on the log_level argument
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
        force=True
    )


def classify_documents(input_docs_dir: str, embedding_model_path: str = None, classifier_model_path: str = None,
                       embedding_model=None, classifier_model=None):
    if embedding_model is None:
        embedding_model = load_embedding_model(model_path=embedding_model_path)
    if classifier_model is None:
        classifier_model = joblib.load(classifier_model_path)
    X = []
    files_loaded = []
    valid_embeddings = []

    documents = get_documents(input_docs_dir=input_docs_dir)

    if isinstance(embedding_model, KeyedVectors):
        word_to_index = embedding_model.key_to_index
    else:
        word_to_index = {word: idx for idx, word in enumerate(embedding_model.get_words())}

    for _, (file_path, fulltext, _, _) in enumerate(documents):
        text = remove_stopwords(fulltext)
        text = text.lower()
        doc_embedding = get_document_embedding(embedding_model, text, word_to_index=word_to_index)
        X.append(doc_embedding)
        files_loaded.append(file_path)
        valid_embeddings.append(not np.all(doc_embedding == np.zeros_like(doc_embedding)))

    del embedding_model
    X = np.array(X)
    classifications = classifier_model.predict(X)
    try:
        confidence_scores = [classes_proba[1] for classes_proba in classifier_model.predict_proba(X)]
    except AttributeError:
        confidence_scores = [1 / (1 + np.exp(-decision_value)) for decision_value in classifier_model.decision_function(X)]
    return files_loaded, classifications, confidence_scores, valid_embeddings


def parse_arguments():
    parser = argparse.ArgumentParser(description='Classify documents')
    parser.add_argument("-e", "--embedding_model_path", type=str, help="Path to the word embedding model")
    parser.add_argument("-l", "--log_level", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="Set the logging level")
    parser.add_argument("-f", "--reference_curie", type=str, help="Only run for this references.", required=False)
    parser.add_argument("-m", "--mod_abbreviation", type=str, help="Only run for this mod.", required=False)
    parser.add_argument("-t", "--topic", type=str, help="Only run for this topic.", required=False)
    parser.add_argument("-s", "--stage", action="store_true", help="Only run for on stage.", required=False)
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode (directly on provided"
                                                                 " references and mod - do not store results nor send "
                                                                 "emails).", required=False)

    return parser.parse_args()


def process_classification_jobs(mod_id, topic, jobs, embedding_model, test_mode=False):
    mod_abbr = get_cached_mod_abbreviation_from_id(mod_id)
    tet_source_id = get_tet_source_id(mod_abbreviation=mod_abbr, source_method="abc_document_classifier",
                                      source_description="Alliance document classification pipeline using machine "
                                                         "learning to identify papers of interest for curation data "
                                                         "types")
    classifier_file_path = (f"/data/agr_document_classifier/biocuration_topic_classification_{mod_abbr}_"
                            f"{topic.replace(':', '_')}_classifier.joblib")
    if test_mode and os.path.exists(classifier_file_path):
        logger.info(f"Test mode: using existing classifier model for mod: {mod_abbr}, topic: {topic}.")
    else:
        try:
            download_abc_model(mod_abbreviation=mod_abbr, topic=topic, output_path=classifier_file_path,
                               task_type="biocuration_topic_classification")
            logger.info(f"Classification model downloaded for mod: {mod_abbr}, topic: {topic}.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Classification model not found for mod: {mod_abbr}, topic: {topic}. Skipping.")
                return
            else:
                raise
    try:
        model_meta_data = get_model_data(mod_abbreviation=mod_abbr, task_type="biocuration_topic_classification",
                                         topic=topic)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"ml_model data not found for mod: {mod_abbr}, topic: {topic}. Skipping.")
            return
        else:
            raise

    classification_batch_size = int(os.environ.get("CLASSIFICATION_BATCH_SIZE", 1000))
    jobs_to_process = copy.deepcopy(jobs)
    classifier_model = joblib.load(classifier_file_path)
    while jobs_to_process:
        if os.path.isfile('/data/agr_document_classifier/stop_classifier'):
            logger.info("Stopping classifier due to time limit (stop file exists)")
            return
        job_batch = jobs_to_process[:classification_batch_size]
        jobs_to_process = jobs_to_process[classification_batch_size:]
        logger.info(f"Processing a batch of {str(len(job_batch))} jobs. "
                    f"Jobs remaining to process: {str(len(jobs_to_process))}")
        process_job_batch(job_batch, mod_abbr, topic, tet_source_id, embedding_model, classifier_model, model_meta_data,
                          test_mode)


def process_job_batch(job_batch, mod_abbr, topic, tet_source_id, embedding_model, classifier_model, model_meta_data,
                      test_mode):
    reference_curie_job_map = {job["reference_curie"]: job for job in job_batch}
    prepare_classification_directory()
    download_tei_files_for_references(list(reference_curie_job_map.keys()),
                                      "/data/agr_document_classifier/to_classify", mod_abbr)
    files_loaded, classifications, conf_scores, valid_embeddings = classify_documents(
        embedding_model=embedding_model,
        classifier_model=classifier_model,
        input_docs_dir="/data/agr_document_classifier/to_classify")
    if test_mode:
        for file_path, classification, conf_score, valid_embedding in zip(files_loaded, classifications, conf_scores,
                                                                          valid_embeddings):
            reference_curie = file_path.split("/")[-1].replace("_", ":")[:-4]
            if not valid_embedding:
                logger.warning(f"Invalid embedding for file: {file_path}. Skipping.")
                continue
            confidence_level = get_confidence_level(classification, conf_score)
            logger.info(f"reference_curie: '{reference_curie}', species: '{mod_abbr}', topic: '{topic}', "
                        f"classification: '{classification}', confidence_score: '{conf_score}', "
                        f"confidence_level: '{confidence_level}', tet_source_id: '{tet_source_id}' "
                        f"data_novelty: '{model_meta_data['data_novelty']}'")
        logger.info(f"Finished processing batch of {len(files_loaded)} jobs in test mode. Positive: "
                    f"{sum(classifications)}. Negative: {len(classifications)-sum(classifications)}")
    else:
        send_classification_results(files_loaded, classifications, conf_scores, valid_embeddings,
                                    reference_curie_job_map, mod_abbr, topic, tet_source_id, model_meta_data)


def prepare_classification_directory():
    os.makedirs("/data/agr_document_classifier/to_classify", exist_ok=True)
    logger.info("Cleaning up existing files in the to_classify directory")
    for file in os.listdir("/data/agr_document_classifier/to_classify"):
        os.remove(os.path.join("/data/agr_document_classifier/to_classify", file))


def send_classification_results(files_loaded, classifications, conf_scores, valid_embeddings, reference_curie_job_map,
                                mod_abbr, topic, tet_source_id, model_meta_data):
    logger.info("Sending classification tags to ABC.")
    species = get_cached_mod_species_map()[mod_abbr]
    if 'species' in model_meta_data and model_meta_data['species'] and model_meta_data['species'].startswith("NCBITaxon:"):
        species = model_meta_data['species']
    send_to_manual_indexing = False
    if topic == 'ATP:0000207' and mod_abbr == 'FB':
        send_to_manual_indexing = True

    for file_path, classification, conf_score, valid_embedding in zip(files_loaded, classifications, conf_scores,
                                                                      valid_embeddings):
        reference_curie = file_path.split("/")[-1].replace("_", ":")[:-4]
        if not valid_embedding:
            logger.warning(f"Invalid embedding for file: {file_path}. Setting job to failed.")
            set_job_started(reference_curie_job_map[reference_curie])
            set_job_failure(reference_curie_job_map[reference_curie])
            continue
        confidence_level = get_confidence_level(classification, conf_score)

        result = True
        if classification > 0 or model_meta_data['negated']:
            logger.debug(f"reference_curie: '{reference_curie}', species: '{species}', topic: '{topic}', confidence_level: '{confidence_level}', tet_source_id: '{tet_source_id}' data_novelty: '{model_meta_data['data_novelty']}")
            if send_to_manual_indexing:
                result = send_manual_indexing_to_abc(reference_curie, mod_abbr, topic, conf_score)
            else:
                result = send_classification_tag_to_abc(
                    reference_curie, species, topic,
                    negated=bool(classification == 0),
                    data_novelty=model_meta_data['data_novelty'],
                    confidence_score=conf_score,
                    confidence_level=confidence_level,
                    tet_source_id=tet_source_id,
                    ml_model_id=model_meta_data['ml_model_id'])
        if result:
            set_job_success(reference_curie_job_map[reference_curie])
        os.remove(file_path)
    logger.info(f"Finished processing batch of {len(files_loaded)} jobs.")


def get_confidence_level(classification, conf_score):
    if classification == 0:
        return "NEG"
    elif conf_score < 0.667:
        return "LOW"
    elif conf_score < 0.833:
        return "MEDIUM"
    else:
        return "HIGH"


def classify_mode(args: Namespace):
    logger.info("Classification started.")

    mod_topic_jobs = load_all_jobs("classification_job", args)
    embedding_model = load_embedding_model(args.embedding_model_path)
    failed_processes = []
    for (mod_id, topic), jobs in mod_topic_jobs.items():
        try:
            process_classification_jobs(mod_id, topic, jobs, embedding_model)
        except Exception as e:
            logger.error(f"Error processing a batch of '{topic}' jobs for {mod_id}: {e}")
            failed = {'topic': topic,
                      'mod_abbreviation': mod_id,
                      'exception': str(e)}
            formatted_traceback = traceback.format_tb(e.__traceback__)
            failed['trace'] = ""
            for line in formatted_traceback:
                failed['trace'] += f"{line}<br>"
            failed_processes.append(failed)
        if os.path.isfile('/data/agr_document_classifier/stop_classifier'):
            logger.info("Stopping classifier due to time limit (stop file exists)")
            os.remove('/data/agr_document_classifier/stop_classifier')
            break

    if failed_processes:
        subject = "Failed processing of classification jobs"
        message = "<h>The following jobs failed to process:</h><br><br>\n\n"
        for fp in failed_processes:
            message += f"Topic: {fp['topic']}  mod_id:{fp['mod_abbreviation']}<br>\n"
            message += f"Exception: {fp['exception']}<br>\n"
            message += f"Stacktrace: {fp['trace']}<br><br>\n\n"
        send_report(subject, message)
        exit(-1)


def direct_classify_mode(args: Namespace):
    logger.info(f"Direct classification started for mod={args.mod_abbreviation}, topic={args.topic}")

    if not args.reference_curie:
        logger.error("No references provided for direct classification.")
        return

    reference_curies = [ref_curie.strip() for ref_curie in args.reference_curie.split(",")]
    if not reference_curies:
        logger.error("No valid reference curies found in input.")
        return

    # Build fake jobs in the same shape as load_all_jobs would return
    jobs = [{"reference_curie": ref} for ref in reference_curies]

    embedding_model = load_embedding_model(args.embedding_model_path)

    try:
        process_classification_jobs(
            mod_id=get_cached_mod_id_from_abbreviation(args.mod_abbreviation),
            topic=args.topic,
            jobs=jobs,
            embedding_model=embedding_model,
            test_mode=True
        )
    except Exception as e:
        logger.error(f"Error in direct classification: {e}")
        formatted_traceback = traceback.format_tb(e.__traceback__)
        message = f"Direct classification failed for references {reference_curies}<br>"
        message += f"Exception: {str(e)}<br>\nStacktrace:<br>{''.join(formatted_traceback)}"
        send_report("Direct classification failed", message)
        exit(-1)


def main():
    args: Namespace = parse_arguments()
    configure_logging(args.log_level)
    if args.test_mode:
        direct_classify_mode(args)
    else:
        if args.stage:
            set_blue_api_base_url("https://stage-literature-rest.alliancegenome.org")
            os.environ['ABC_API_SERVER'] = "https://stage-literature-rest.alliancegenome.org"
            os.environ["ON_PRODUCTION"] = "no"
        else:
            os.environ["ON_PRODUCTION"] = "yes"
        classify_mode(args)


if __name__ == '__main__':
    main()
