import argparse
import copy
import logging
import os
import os.path
import sys

import traceback

import joblib
import numpy as np
import requests.exceptions
from gensim.models import KeyedVectors


from utils.abc_utils import download_tei_files_for_references, send_classification_tag_to_abc, \
    get_cached_mod_abbreviation_from_id, \
    set_job_success, get_tet_source_id, set_job_started, \
    download_abc_model, set_job_failure, load_all_jobs
from utils.get_documents import get_documents
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
        doc_embedding = get_document_embedding(embedding_model, fulltext, word_to_index=word_to_index)
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
    return parser.parse_args()


def process_classification_jobs(mod_id, topic, jobs, embedding_model):
    mod_abbr = get_cached_mod_abbreviation_from_id(mod_id)
    tet_source_id = get_tet_source_id(mod_abbreviation=mod_abbr, source_method="abc_document_classifier",
                                      source_description="Alliance document classification pipeline using machine "
                                                         "learning to identify papers of interest for curation data "
                                                         "types")
    classifier_file_path = (f"/data/agr_document_classifier/biocuration_topic_classification_{mod_abbr}_"
                            f"{topic.replace(':', '_')}_classifier.joblib")
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
    classification_batch_size = int(os.environ.get("CLASSIFICATION_BATCH_SIZE", 1000))
    jobs_to_process = copy.deepcopy(jobs)
    classifier_model = joblib.load(classifier_file_path)
    while jobs_to_process:
        if os.path.isfile('/data/agr_document_classifier/stop_classifier'):
            logger.info("Stopping classifier due to time limit (stop file exists)")
            return
        job_batch = jobs_to_process[:classification_batch_size]
        jobs_to_process = jobs_to_process[classification_batch_size:]
        logger.info(f"Processing a batch of {str(classification_batch_size)} jobs. "
                    f"Jobs remaining to process: {str(len(jobs_to_process))}")
        process_job_batch(job_batch, mod_abbr, topic, tet_source_id, embedding_model, classifier_model)


def process_job_batch(job_batch, mod_abbr, topic, tet_source_id, embedding_model, classifier_model):
    reference_curie_job_map = {job["reference_curie"]: job for job in job_batch}
    prepare_classification_directory()
    download_tei_files_for_references(list(reference_curie_job_map.keys()),
                                      "/data/agr_document_classifier/to_classify", mod_abbr)
    files_loaded, classifications, conf_scores, valid_embeddings = classify_documents(
        embedding_model=embedding_model,
        classifier_model=classifier_model,
        input_docs_dir="/data/agr_document_classifier/to_classify")
    send_classification_results(files_loaded, classifications, conf_scores, valid_embeddings, reference_curie_job_map,
                                mod_abbr, topic, tet_source_id)


def prepare_classification_directory():
    os.makedirs("/data/agr_document_classifier/to_classify", exist_ok=True)
    logger.info("Cleaning up existing files in the to_classify directory")
    for file in os.listdir("/data/agr_document_classifier/to_classify"):
        os.remove(os.path.join("/data/agr_document_classifier/to_classify", file))


def send_classification_results(files_loaded, classifications, conf_scores, valid_embeddings, reference_curie_job_map,
                                mod_abbr, topic, tet_source_id):
    logger.info("Sending classification tags to ABC.")
    for file_path, classification, conf_score, valid_embedding in zip(files_loaded, classifications, conf_scores,
                                                                      valid_embeddings):
        reference_curie = file_path.split("/")[-1].replace("_", ":")[:-4]
        if not valid_embedding:
            logger.warning(f"Invalid embedding for file: {file_path}. Setting job to failed.")
            set_job_started(reference_curie_job_map[reference_curie])
            set_job_failure(reference_curie_job_map[reference_curie])
            continue
        confidence_level = get_confidence_level(classification, conf_score)
        result = send_classification_tag_to_abc(reference_curie, mod_abbr, topic,
                                                negated=bool(classification == 0),
                                                confidence_level=confidence_level, tet_source_id=tet_source_id)
        if result:
            # No need to set started and then immediately set to completed
            # The transition table should have both needed and in progress going to completed
            # set_job_started(reference_curie_job_map[reference_curie])
            set_job_success(reference_curie_job_map[reference_curie])
        os.remove(file_path)
    logger.info(f"Finished processing batch of {len(files_loaded)} jobs.")


def get_confidence_level(classification, conf_score):
    if classification == 0:
        return "NEG"
    elif conf_score < 0.5:
        return "Low"
    elif conf_score < 0.75:
        return "Med"
    else:
        return "High"


def classify_mode(args):
    logger.info("Classification started.")
    print("#############")
    logger.debug("deb")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")

    mod_topic_jobs = load_all_jobs("classification_job")
    embedding_model = load_embedding_model(args.embedding_model_path)
    failed_processes = []
    for (mod_id, topic), jobs in mod_topic_jobs.items():
        try:
            process_classification_jobs(mod_id, topic, jobs, embedding_model)
        except Exception as e:
            logger.error(f"Error processing a batch of '{topic}' jobs for {mod_id}.")
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


def main():
    args = parse_arguments()
    configure_logging(args.log_level)

    classify_mode(args)


if __name__ == '__main__':
    main()
