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
import scipy.sparse as sp
from gensim.models import KeyedVectors

from utils.abc_utils import download_md_files_for_references, send_classification_tag_to_abc, \
    get_cached_mod_abbreviation_from_id, \
    set_job_success, get_tet_source_id, set_job_started, \
    download_abc_model, set_job_failure, load_all_jobs, get_model_data, \
    set_blue_api_base_url, \
    get_cached_mod_id_from_abbreviation, send_manual_indexing_to_abc, create_workflow_tag, \
    get_current_workflow_status, get_reference_embedding
from utils.abc_embeddings import is_abc_embedding_model, ABC_EMBEDDING_DIM
from utils.get_documents import get_documents, remove_stopwords
from utils.embedding import load_embedding_model, build_document_features, get_bow_vectorizer

from agr_literature_service.lit_processing.utils.report_utils import send_report
from utils.slack_utils import send_slack_notification, format_skipped_jobs_html


logger = logging.getLogger(__name__)

# SCRUM-6203: an FB "no genetic data" (ATP:0000207) paper only gets the "manual indexing
# status TBD" workflow tag (ATP:0000359) -- i.e. only surfaces for manual indexing -- when the
# classifier's positive-class confidence is at least this threshold. Previously the tag was
# created for every classified paper (an effective threshold of 0); it was raised now that a
# weekly pipeline exports every manual_indexing_tag score to FB directly, so the low-confidence
# papers no longer need surfacing. The manual_indexing_tag row itself is still written for every
# paper regardless of confidence (the export consumes all of them).
FB_NO_GENETIC_DATA_TBD_THRESHOLD = 0.5


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
                       embedding_model=None, classifier_model=None,
                       use_bow_features: bool = False, use_max_pooling: bool = False,
                       use_lsh_features: bool = False,
                       include_keywords: bool = False, include_metadata: bool = False):
    if embedding_model is None:
        embedding_model = load_embedding_model(model_path=embedding_model_path)
    if classifier_model is None:
        classifier_model = joblib.load(classifier_model_path)
    rows = []
    files_loaded = []
    valid_embeddings = []

    documents = get_documents(input_docs_dir=input_docs_dir, include_keywords=include_keywords,
                              include_metadata=include_metadata)

    if isinstance(embedding_model, KeyedVectors):
        word_to_index = embedding_model.key_to_index
        embedding_dim = embedding_model.vector_size
    else:
        word_to_index = {word: idx for idx, word in enumerate(embedding_model.get_words())}
        embedding_dim = embedding_model.get_dimension()
    bow_vectorizer = get_bow_vectorizer() if use_bow_features else None
    # The LSH and BoW blocks both make the feature row sparse.
    sparse_features = use_bow_features or use_lsh_features

    for _, (file_path, fulltext, _, _) in enumerate(documents):
        text = remove_stopwords(fulltext)
        text = text.lower()
        features = build_document_features(embedding_model, text, use_max_pooling=use_max_pooling,
                                           use_bow=use_bow_features, use_lsh=use_lsh_features,
                                           word_to_index=word_to_index, bow_vectorizer=bow_vectorizer)
        rows.append(features)
        files_loaded.append(file_path)
        # The "valid embedding" gate stays tied to the mean-embedding block (first
        # embedding_dim dims); the optional max/LSH/BoW blocks do not change which
        # documents are considered classifiable.
        mean_block = features[0, :embedding_dim].toarray().ravel() if sparse_features else features[:embedding_dim]
        valid_embeddings.append(not np.all(mean_block == 0))

    del embedding_model
    if not files_loaded:
        logger.warning(
            "classify_documents called with zero loaded documents for input_docs_dir=%s; "
            "skipping predict()", input_docs_dir,
        )
        return files_loaded, np.array([]), [], valid_embeddings
    X = sp.vstack(rows, format="csr") if sparse_features else np.vstack(rows)
    classifications, confidence_scores = predict_labels_and_confidence(classifier_model, X)
    return files_loaded, classifications, confidence_scores, valid_embeddings


def classify_documents_from_abc_embeddings(reference_curies, mod_abbr, classifier_model, use_bow=False,
                                           embedding_cache=None):
    """Classify references using the ABC precomputed embeddings (SCRUM-5781).

    The per-reference dense feature is the L2-normalized chunk-mean pool of the
    main-PDF paragraph embeddings (:func:`utils.abc_utils.get_reference_embedding`);
    when ``use_bow`` is set the hashed BoW block over the parquet's paragraph text
    is concatenated, exactly as at train time (the model's metadata marker says
    which). Returns the same 4-tuple shape as :func:`classify_documents` —
    ``(ids_loaded, classifications, confidence_scores, valid_embeddings)`` — but
    ``ids_loaded`` holds reference curies rather than file paths (downstream
    recovers the curie from either).

    A reference with no available embedding is kept as a zero row flagged invalid,
    so the caller fails that job just like a missing MD in the BioWordVec path.

    ``embedding_cache`` is an optional ``{curie: (pooled, text) | None}`` dict; when
    provided the ABC embedding is fetched at most once per reference across calls,
    which matters when several models are applied to the same references (the
    re-classification pipeline shares one cache over all topics).
    """
    bow_vectorizer = get_bow_vectorizer() if use_bow else None
    rows = []
    ids_loaded = []
    valid_embeddings = []
    for reference_curie in reference_curies:
        if embedding_cache is not None and reference_curie in embedding_cache:
            result = embedding_cache[reference_curie]
        else:
            result = get_reference_embedding(reference_curie, mod_abbr)
            if embedding_cache is not None:
                embedding_cache[reference_curie] = result
        ids_loaded.append(reference_curie)
        if result is None:
            pooled, text = np.zeros(ABC_EMBEDDING_DIM, dtype=np.float32), ""
            valid_embeddings.append(False)
        else:
            pooled, text = result
            valid_embeddings.append(True)
        if use_bow:
            bow = bow_vectorizer.transform([remove_stopwords(text).lower() if text else ""])
            rows.append(sp.hstack([sp.csr_matrix(pooled.reshape(1, -1)), bow], format="csr"))
        else:
            rows.append(pooled)
    if not ids_loaded:
        return ids_loaded, np.array([]), [], valid_embeddings
    X = sp.vstack(rows, format="csr") if use_bow else np.vstack(rows)
    classifications, confidence_scores = predict_labels_and_confidence(classifier_model, X)
    return ids_loaded, classifications, confidence_scores, valid_embeddings


def predict_labels_and_confidence(classifier_model, X):
    """Predict labels together with a confidence score for the positive class.

    ``confidence_score`` is the probability of the positive class
    (``classifier_model.classes_[1]``). The label is derived from that score
    (``score >= 0.5`` -> positive) instead of from ``classifier_model.predict()``
    so the two can never disagree.

    This matters for ``SVC(probability=True)``: its ``predict()`` uses the sign of
    the raw SVM margin while ``predict_proba()`` uses a separately fitted
    Platt-scaling model, and near the decision boundary the two disagree, yielding
    a positive classification with ``confidence_score < 0.5`` (documented
    scikit-learn behavior). For every other classifier ``predict()`` already
    equals ``argmax(predict_proba())``, so deriving the label from the score is a
    no-op.
    """
    try:
        confidence_scores = [classes_proba[1] for classes_proba in classifier_model.predict_proba(X)]
    except AttributeError:
        # Models without predict_proba (e.g. LinearSVC): map the signed margin
        # through a sigmoid. predict() is sign(decision_function) there, so label
        # and score are already consistent; we derive both the same way anyway.
        confidence_scores = [1 / (1 + np.exp(-decision_value))
                             for decision_value in classifier_model.decision_function(X)]
    negative_label, positive_label = classifier_model.classes_[0], classifier_model.classes_[1]
    classifications = np.array([positive_label if score >= 0.5 else negative_label
                                for score in confidence_scores])
    return classifications, confidence_scores


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
    parser.add_argument("--use_bow_features", action="store_true",
                        help="Concatenate a stateless hashing bag-of-words block with the embedding. "
                             "Must match the flag used when the model was trained.", required=False)
    parser.add_argument("--use_max_pooling", action="store_true",
                        help="Concatenate an element-wise max-pooled embedding alongside the mean. "
                             "Must match the flag used when the model was trained.", required=False)
    parser.add_argument("--use_lsh_features", action="store_true",
                        help="Concatenate a stateless LSH bag-of-concepts block (random-hyperplane "
                             "buckets over the word embeddings). "
                             "Must match the flag used when the model was trained.", required=False)
    parser.add_argument("--include_keywords", action="store_true",
                        help="Include author keywords in the document text. "
                             "Must match the flag used when the model was trained.", required=False)
    parser.add_argument("--include_metadata", action="store_true",
                        help="Include article metadata in the document text. "
                             "Must match the flag used when the model was trained.", required=False)

    return parser.parse_args()


def process_classification_jobs(mod_id, topic, jobs, embedding_model, test_mode=False,
                                use_bow_features=False, use_max_pooling=False, use_lsh_features=False,
                                include_keywords=False, include_metadata=False):
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
                return {"mod_abbreviation": mod_abbr, "topic": topic, "jobs": len(jobs),
                        "reason": "classification model not found"}
            else:
                raise
    try:
        model_meta_data = get_model_data(mod_abbreviation=mod_abbr, task_type="biocuration_topic_classification",
                                         topic=topic)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"ml_model data not found for mod: {mod_abbr}, topic: {topic}. Skipping.")
            return {"mod_abbreviation": mod_abbr, "topic": topic, "jobs": len(jobs),
                    "reason": "ml_model data not found"}
        else:
            raise

    # Retrocompat switch (SCRUM-5781): a model trained on ABC embeddings has the
    # embedding_* columns populated in its ml_model metadata. When set we fetch ABC
    # embeddings for it; when absent (legacy BioWordVec model) the on-the-fly text
    # path below runs unchanged.
    use_abc_embeddings = is_abc_embedding_model(model_meta_data)
    # Pooling (L2 chunk-mean) and the BoW block are fixed conventions applied to
    # every ABC-embedding model, so they are not stored on the model — always on.
    abc_use_bow = True
    if use_abc_embeddings:
        logger.info(f"Model for mod: {mod_abbr}, topic: {topic} uses ABC embeddings "
                    f"(profile {model_meta_data.get('embedding_profile')} "
                    f"v{model_meta_data.get('embedding_version')}).")

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
                          test_mode, use_bow_features=use_bow_features, use_max_pooling=use_max_pooling,
                          use_lsh_features=use_lsh_features,
                          include_keywords=include_keywords, include_metadata=include_metadata,
                          use_abc_embeddings=use_abc_embeddings, abc_use_bow=abc_use_bow)


def process_job_batch(job_batch, mod_abbr, topic, tet_source_id, embedding_model, classifier_model, model_meta_data,
                      test_mode, use_bow_features=False, use_max_pooling=False, use_lsh_features=False,
                      include_keywords=False, include_metadata=False, use_abc_embeddings=False,
                      abc_use_bow=False):
    reference_curie_job_map = {job["reference_curie"]: job for job in job_batch}
    if use_abc_embeddings:
        # ABC-embedding models classify from the precomputed vectors directly — no
        # MD download and no BioWordVec/max/LSH blocks. The BoW block (if the model
        # was trained with it, per its marker) is hashed from the parquet's own
        # paragraph text. References without an embedding come back flagged invalid
        # and are failed in the sender below, matching the missing-MD policy.
        files_loaded, classifications, conf_scores, valid_embeddings = classify_documents_from_abc_embeddings(
            list(reference_curie_job_map.keys()), mod_abbr, classifier_model, use_bow=abc_use_bow)
    else:
        prepare_classification_directory()
        download_md_files_for_references(list(reference_curie_job_map.keys()),
                                         "/data/agr_document_classifier/to_classify", mod_abbr)
        expected_curies = set(reference_curie_job_map.keys())
        downloaded_curies = {
            os.path.splitext(f)[0].replace("_", ":")
            for f in os.listdir("/data/agr_document_classifier/to_classify")
            if f.endswith(".md") and ".supp_" not in f
        }
        missing_curies = expected_curies - downloaded_curies
        if missing_curies:
            logger.warning(
                "No MD available in ABC for %d/%d references in this batch (mod=%s topic=%s); "
                "marking those jobs failed",
                len(missing_curies), len(expected_curies), mod_abbr, topic,
            )
            if not test_mode:
                for curie in missing_curies:
                    job = reference_curie_job_map[curie]
                    set_job_started(job)
                    set_job_failure(job)
        files_loaded, classifications, conf_scores, valid_embeddings = classify_documents(
            embedding_model=embedding_model,
            classifier_model=classifier_model,
            input_docs_dir="/data/agr_document_classifier/to_classify",
            use_bow_features=use_bow_features,
            use_max_pooling=use_max_pooling,
            use_lsh_features=use_lsh_features,
            include_keywords=include_keywords,
            include_metadata=include_metadata)
    if test_mode:
        for file_path, classification, conf_score, valid_embedding in zip(files_loaded, classifications, conf_scores,
                                                                          valid_embeddings):
            reference_curie = os.path.splitext(file_path.split("/")[-1])[0].replace("_", ":")
            if not valid_embedding:
                logger.warning(f"Invalid embedding for file: {file_path}. Skipping.")
                continue
            confidence_level = get_confidence_level(classification, conf_score)
            logger.info(f"reference_curie: '{reference_curie}', species: '{mod_abbr}', topic: '{topic}', "
                        f"classification: '{classification}', confidence_score: '{conf_score}', "
                        f"confidence_level: '{confidence_level}', tet_source_id: '{tet_source_id}' "
                        f"data_novelty: '{model_meta_data['data_novelty']}'")
        logger.info(f"Finished processing batch of {len(files_loaded)} jobs in test mode. Positive: "
                    f"{sum(classifications)}. Negative: {len(classifications) - sum(classifications)}")
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
    species = None
    if 'species' in model_meta_data and model_meta_data['species'] and model_meta_data['species'].startswith("NCBITaxon:"):
        species = model_meta_data['species']
    send_to_manual_indexing = False
    if topic == 'ATP:0000207' and mod_abbr == 'FB':
        send_to_manual_indexing = True

    for file_path, classification, conf_score, valid_embedding in zip(files_loaded, classifications, conf_scores,
                                                                      valid_embeddings):
        reference_curie = os.path.splitext(file_path.split("/")[-1])[0].replace("_", ":")
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
                # SCRUM-6203: only set the "manual indexing status TBD" workflow tag for FB
                # "no genetic data" when the model is actually positive for the topic, i.e.
                # confidence_score >= 0.5. The manual_indexing_tag above is still written for
                # every paper (the weekly FB export consumes all scores).
                if (result and topic == 'ATP:0000207'
                        and conf_score is not None
                        and conf_score >= FB_NO_GENETIC_DATA_TBD_THRESHOLD):
                    # Only create workflow tag if no manual indexing workflow tag exists
                    current_manual_indexing_status = get_current_workflow_status(
                        reference_curie, mod_abbr, "ATP:0000273")
                    if current_manual_indexing_status is None:
                        create_workflow_tag(reference_curie=reference_curie, mod_abbreviation=mod_abbr,
                                            workflow_tag_atp_id="ATP:0000359")
                    else:
                        logger.debug(f"Skipping workflow tag creation for {reference_curie}: "
                                     f"manual indexing workflow tag already exists "
                                     f"({current_manual_indexing_status})")
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
        # Legacy path writes a real MD file to clean up here; the ABC path passes
        # reference curies (no file on disk), so only remove when it exists.
        if os.path.exists(file_path):
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
    skipped_jobs = []
    for (mod_id, topic), jobs in mod_topic_jobs.items():
        try:
            skip = process_classification_jobs(mod_id, topic, jobs, embedding_model,
                                               use_bow_features=args.use_bow_features,
                                               use_max_pooling=args.use_max_pooling,
                                               use_lsh_features=args.use_lsh_features,
                                               include_keywords=args.include_keywords,
                                               include_metadata=args.include_metadata)
            if skip:
                skipped_jobs.append(skip)
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

    if failed_processes or skipped_jobs:
        if failed_processes and skipped_jobs:
            subject = "Failed and skipped classification jobs"
        elif skipped_jobs:
            subject = "Skipped classification jobs (missing model)"
        else:
            subject = "Failed processing of classification jobs"
        message = ""
        if failed_processes:
            message += "<h>The following jobs failed to process:</h><br><br>\n\n"
            for fp in failed_processes:
                message += f"Topic: {fp['topic']}  mod_id:{fp['mod_abbreviation']}<br>\n"
                message += f"Exception: {fp['exception']}<br>\n"
                message += f"Stacktrace: {fp['trace']}<br><br>\n\n"
        message += format_skipped_jobs_html(skipped_jobs)
        send_report(subject, message)
        send_slack_notification(subject, message)
        if failed_processes:
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
            test_mode=True,
            use_bow_features=args.use_bow_features,
            use_max_pooling=args.use_max_pooling,
            use_lsh_features=args.use_lsh_features,
            include_keywords=args.include_keywords,
            include_metadata=args.include_metadata
        )
    except Exception as e:
        logger.error(f"Error in direct classification: {e}")
        formatted_traceback = traceback.format_tb(e.__traceback__)
        message = f"Direct classification failed for references {reference_curies}<br>"
        message += f"Exception: {str(e)}<br>\nStacktrace:<br>{''.join(formatted_traceback)}"
        send_report("Direct classification failed", message)
        send_slack_notification("Direct classification failed", message)
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
