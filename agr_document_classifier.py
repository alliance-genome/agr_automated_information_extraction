import argparse
import glob
import json
import logging
import os
import os.path
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List

import fasttext
import joblib
import nltk
import numpy as np
from gensim.models import KeyedVectors
from grobid_client import Client
from grobid_client.api.pdf import process_fulltext_document
from grobid_client.models import Article, ProcessForm, TextWithRefs
from grobid_client.types import TEI, File
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from abc_utils import get_jobs_to_classify, download_tei_files_for_references, get_curie_from_reference_id, \
    send_classification_tag_to_abc, get_cached_mod_abbreviation_from_id, \
    job_category_topic_map, set_job_success, get_tet_source_id, set_job_started, get_training_set_from_abc, \
    upload_classification_model, download_classification_model
from dataset_downloader import download_tei_files_from_abc_or_convert_pdf
from models import POSSIBLE_CLASSIFIERS

nltk.download('stopwords')
nltk.download('punkt')

# Configure the logging in the main script
logger = logging.getLogger(__name__)


def get_document_embedding(model, document, weighted_average_word_embedding: bool = False,
                           standardize_embeddings: bool = False, normalize_embeddings: bool = False,
                           word_to_index=None):
    # Split the document into words
    words = document.split()
    if isinstance(model, KeyedVectors):
        vocab = set(model.key_to_index.keys())
        valid_words = [word for word in words if word in vocab]
        embeddings = model[valid_words]
        if word_to_index is None:
            word_to_index = model.key_to_index
    else:
        vocab = set(model.get_words())
        valid_words = [word for word in words if word in vocab]
        embeddings = np.array([model.get_word_vector(word) for word in valid_words])
        if word_to_index is None:
            word_to_index = {word: idx for idx, word in enumerate(model.get_words())}

    if embeddings.size == 0:
        return np.zeros(model.get_dimension())

    epsilon = 1e-10
    embeddings_2d = embeddings

    if standardize_embeddings:
        # Standardize the embeddings
        scaler = StandardScaler()
        embeddings_2d = scaler.fit_transform(embeddings_2d)

    if normalize_embeddings:
        # Normalize the embeddings
        norm = np.linalg.norm(embeddings_2d, axis=1, keepdims=True) + epsilon
        embeddings_2d /= norm

    if weighted_average_word_embedding:
        weights = np.array([word_to_index[word] / len(word_to_index) for word in valid_words])
        doc_embedding = np.average(embeddings_2d, axis=0, weights=weights)
    else:
        doc_embedding = np.mean(embeddings_2d, axis=0)
    return doc_embedding


def load_embedding_model(model_path):
    logger.info("Loading embeddings...")
    if model_path.endswith(".vec.bin"):
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    else:
        model = fasttext.load_model(model_path)
    logger.info("Finished loading embeddings.")
    return model


def train_classifier(embedding_model_path: str, training_data_dir: str, weighted_average_word_embedding: bool = False,
                     standardize_embeddings: bool = False, normalize_embeddings: bool = False,
                     sections_to_use: List[str] = None):
    embedding_model = load_embedding_model(model_path=embedding_model_path)

    X = []
    y = []

    # Precompute word_to_index
    if isinstance(embedding_model, KeyedVectors):
        word_to_index = embedding_model.key_to_index
    else:
        word_to_index = {word: idx for idx, word in enumerate(embedding_model.get_words())}

    # For each document in your training data, extract embeddings and labels
    logger.info("Loading training set")
    for label in ["positive", "negative"]:
        documents = list(get_documents(os.path.join(training_data_dir, label, "*")))

        for idx, (_, fulltext, title, abstract) in enumerate(documents, start=1):
            text = ""
            if not sections_to_use:
                text = fulltext
            else:
                if "title" in sections_to_use:
                    text = title
                if "fulltext" in sections_to_use:
                    text += " " + fulltext
                if "abstract" in sections_to_use:
                    text += " " + abstract
            if text:
                text = remove_stopwords(text)
                text = text.lower()
                text_embedding = get_document_embedding(embedding_model, text,
                                                        weighted_average_word_embedding=weighted_average_word_embedding,
                                                        standardize_embeddings=standardize_embeddings,
                                                        normalize_embeddings=normalize_embeddings,
                                                        word_to_index=word_to_index)
                X.append(text_embedding)
                y.append(int(label == "positive"))

    del embedding_model
    logger.info("Finished loading training set.")
    logger.info(f"Dataset size: {str(len(X))}")

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    best_score = 0
    best_classifier = None
    best_params = None
    best_classifier_name = ""
    best_results = {}
    best_index = 0

    stratified_k_folds = StratifiedKFold(n_splits=5)

    scoring = {
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    logger.info("Starting model selection with hyperparameter optimization and cross-validation.")
    for classifier_name, classifier_info in POSSIBLE_CLASSIFIERS.items():
        logger.info(f"Evaluating model {classifier_name}.")
        random_search = RandomizedSearchCV(estimator=classifier_info['model'], n_iter=100,
                                           param_distributions=classifier_info['params'], cv=stratified_k_folds,
                                           scoring=scoring, refit='f1', verbose=1, n_jobs=-1)
        random_search.fit(X, y)

        logger.info(f"Finished training model and fitting best hyperparameters for {classifier_name}. F1 score: "
                    f"{str(random_search.best_score_)}")

        if random_search.best_score_ > best_score:
            best_score = random_search.best_score_
            best_classifier = random_search.best_estimator_
            best_params = random_search.best_params_
            best_classifier_name = classifier_name
            best_results = random_search.cv_results_
            best_index = random_search.best_index_

    logger.info(f"Selected model {best_classifier_name}.")

    # Retrieve the average precision, recall, and F1 score
    average_precision = best_results['mean_test_precision'][best_index]
    average_recall = best_results['mean_test_recall'][best_index]
    average_f1 = best_results['mean_test_f1'][best_index]

    # Calculate standard deviations
    std_precision = best_results['std_test_precision'][best_index]
    std_recall = best_results['std_test_recall'][best_index]
    std_f1 = best_results['std_test_f1'][best_index]

    stats = {
        "model_name": best_classifier_name,
        "average_precision": round(float(average_precision), 3),
        "average_recall": round(float(average_recall), 3),
        "average_f1": round(float(average_f1), 3),
        "std_precision": round(float(std_precision), 3),
        "std_recall": round(float(std_recall), 3),
        "std_f1": round(float(std_f1), 3),
        "best_params": best_params
    }

    # Return the trained model and performance metrics
    return best_classifier, stats


def save_classifier(classifier, mod_abbreviation: str, topic: str, stats: dict, dataset_id: int):
    model_path = f"/data/agr_document_classifier/training/{mod_abbreviation}_{topic.replace(':', '_')}_classifier.joblib"
    joblib.dump(classifier, model_path)
    upload_classification_model(mod_abbreviation, topic, model_path, stats, dataset_id=dataset_id,
                                file_extension="joblib")


def load_classifier(mod_abbreviation, topic, file_path):
    download_classification_model(mod_abbreviation=mod_abbreviation, topic=topic, output_path=file_path)


def get_sentences_from_tei_section(section):
    sentences = []
    error_count = 0  # Initialize error count
    for paragraph in section.paragraphs:
        if isinstance(paragraph, TextWithRefs):
            paragraph = [paragraph]
        for sentence in paragraph:
            try:
                if not sentence.text.isdigit() and not (
                        len(section.paragraphs) == 3 and
                        section.paragraphs[0][0].text in ['\n', ' '] and
                        section.paragraphs[-1][0].text in ['\n', ' ']
                ):
                    sentences.append(re.sub('<[^<]+>', '', sentence.text))
            except Exception as e:
                error_count += 1
                logger.error(f"Error parsing sentences. Total errors so far for reference: {error_count}")
    sentences = [sentence if sentence.endswith(".") else f"{sentence}." for sentence in sentences]
    return sentences


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)


def get_documents(input_docs_dir: str) -> List[Tuple[str, str, str, str]]:
    documents = []
    client = None
    for file_path in glob.glob(input_docs_dir):
        file_obj = Path(file_path)
        if file_path.endswith(".tei") or file_path.endswith(".pdf"):
            with file_obj.open("rb") as fin:
                if file_path.endswith(".pdf"):
                    if client is None:
                        client = Client(base_url=os.environ.get("GROBID_API_URL"), timeout=1000, verify_ssl=False)
                    logger.info("Started pdf to TEI conversion")
                    form = ProcessForm(
                        segment_sentences="1",
                        input_=File(file_name=file_obj.name, payload=fin, mime_type="application/pdf"))
                    r = process_fulltext_document.sync_detailed(client=client, multipart_data=form)
                    file_stream = r.content
                else:
                    file_stream = fin
                try:
                    article: Article = TEI.parse(file_stream, figures=True)
                except Exception as e:
                    logger.error(f"Error parsing TEI file for {str(file_path)}: {str(e)}")
                    continue
                sentences = []
                for section in article.sections:
                    sentences.extend(get_sentences_from_tei_section(section))
                abstract = ""
                for section in article.sections:
                    if section.name == "ABSTRACT":
                        abstract = " ".join(get_sentences_from_tei_section(section))
                        break
                documents.append((file_path, " ".join(sentences), article.title, abstract))
    return documents


def classify_documents(mod_abbreviation, topic, embedding_model_path: str, classifier_model_path: str, input_docs_dir: str):
    embedding_model = load_embedding_model(model_path=embedding_model_path)
    load_classifier(mod_abbreviation, topic, classifier_model_path)
    classifier_model = joblib.load(classifier_model_path)
    X = []
    files_loaded = []

    documents = get_documents(input_docs_dir=input_docs_dir)

    if isinstance(embedding_model, KeyedVectors):
        word_to_index = embedding_model.key_to_index
    else:
        word_to_index = {word: idx for idx, word in enumerate(embedding_model.get_words())}

    for idx, (file_path, fulltext, title, abstract) in enumerate(documents, start=1):
        doc_embedding = get_document_embedding(embedding_model, fulltext, word_to_index=word_to_index)
        X.append(doc_embedding)
        files_loaded.append(file_path)

    del embedding_model
    X = np.array(X)
    classifications = classifier_model.predict(X)
    confidence_scores = [classes_proba[1] for classes_proba in classifier_model.predict_proba(X)]
    return files_loaded, classifications, confidence_scores


def save_stats_file(stats, file_path, task_type, mod_abbreviation, topic, version_num, file_extension,
                    dataset_id):
    model_data = {
        "task_type": task_type,
        "mod_abbreviation": mod_abbreviation,
        "topic": topic,
        "version_num": version_num,
        "file_extension": file_extension,
        "model_type": stats["model_name"],
        "precision": stats["average_precision"],
        "recall": stats["average_recall"],
        "f1_score": stats["average_f1"],
        "parameters": stats["best_params"],
        "dataset_id": dataset_id
    }
    with open(file_path, "w") as stats_file:
        json.dump(model_data, stats_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify documents or train document classifiers')
    parser.add_argument("-m", "--mode", type=str, choices=['train', 'classify'], default="classify",
                        help="Mode of operation: train or classify")
    parser.add_argument("-d", "--datatype_train", type=str, required=False, help="Datatype to train")
    parser.add_argument("-M", "--mod_train", type=str, required=False, help="MOD to train")
    parser.add_argument("-e", "--embedding_model_path", type=str, help="Path to the word embedding model")
    parser.add_argument("-u", "--sections_to_use", type=str, nargs="+", help="Parts of the articles to use",
                        required=False)
    parser.add_argument("-w", "--weighted_average_word_embedding", action="store_true",
                        help="Whether to use a weighted word embedding based on word frequencies from the model",
                        required=False)
    parser.add_argument("-n", "--normalize_embeddings", action="store_true",
                        help="Whether to normalize the word embedding vectors",
                        required=False)
    parser.add_argument("-s", "--standardize_embeddings", action="store_true",
                        help="Whether to standardize the word embedding vectors",
                        required=False)
    parser.add_argument("-S", "--skip_training_set_download", action="store_true",
                        help="Assume that tei files from training set are already present and do not download them "
                             "again",
                        required=False)
    parser.add_argument("-l", "--log_level", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="Set the logging level")

    args = parser.parse_args()

    # Configure logging based on the log_level argument
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )

    logger = logging.getLogger(__name__)

    if args.mode == "classify":
        mod_datatype_jobs = defaultdict(list)
        limit = 1000
        offset = 0
        jobs_already_added = set()
        logger.info("Loading jobs to classify from ABC ...")

        start_time = time.time()
        last_reported = 0
        total_jobs_estimate = 10000  # Adjust this number if you have an estimate

        while all_jobs := get_jobs_to_classify(limit, offset):
            total_jobs = len(all_jobs)
            for i, job in enumerate(all_jobs, start=1):
                reference_id = job["reference_id"]
                datatype = job["job_name"].replace("_classification_job", "")
                mod_id = job["mod_id"]
                if (mod_id, datatype, reference_id) not in jobs_already_added:
                    mod_datatype_jobs[(mod_id, datatype)].append(job)
                    jobs_already_added.add((mod_id, datatype, reference_id))

                current = offset + i
            offset += limit

        logger.info("Finished loading jobs to classify from ABC ...")

        for (mod_id, datatype), jobs in mod_datatype_jobs.items():
            mod_abbr = get_cached_mod_abbreviation_from_id(mod_id)
            datatype = datatype.replace(" ", "_")
            if datatype != "catalytic_activity" or mod_abbr != "WB":
                continue
            tet_source_id = get_tet_source_id(mod_abbreviation=mod_abbr)
            reference_curie_job_map = {job["reference_curie"]: job for job in jobs}
            os.makedirs("/data/agr_document_classifier/to_classify", exist_ok=True)
            if len(os.listdir("/data/agr_document_classifier/to_classify")) == 0:
                logger.info("Empty file dir. Downloading TEI files from ABC server")
                download_tei_files_for_references(list(reference_curie_job_map.keys()),
                                                  "/data/agr_document_classifier/to_classify", mod_abbr)
            else:
                logger.info("Using existing TEI files")

            topic = job_category_topic_map[datatype]
            files_loaded, classifications, conf_scores = classify_documents(
                mod_abbreviation=args.mod_train, topic=topic,
                embedding_model_path=args.embedding_model_path,
                classifier_model_path=f"/data/agr_document_classifier/{mod_abbr}_{datatype}.joblib",
                input_docs_dir="/data/agr_document_classifier/to_classify")

            total_files = len(files_loaded)
            start_time = time.time()
            last_reported = 0

            for idx, (file_path, classification, conf_score) in enumerate(zip(files_loaded, classifications, conf_scores), start=1):
                confidence_level = "NEG" if classification == 0 else "Low" if conf_score < 0.5 else "Med" if (
                        conf_score < 0.75) else "High"
                reference_curie = file_path.split("/")[-1].replace("_", ":")[:-4]
                result = send_classification_tag_to_abc(reference_curie, mod_abbr, job_category_topic_map[datatype],
                                                        negated=bool(classification == 0),
                                                        confidence_level=confidence_level, tet_source_id=tet_source_id)
                if result is True:
                    set_job_started(reference_curie_job_map[reference_curie])
                    set_job_success(reference_curie_job_map[reference_curie])
                else:
                    # TODO: reset job status to "needs classification"
                    pass
                os.remove(file_path)

    else:
        training_data_dir = "/data/agr_document_classifier/training"
        if args.skip_training_set_download:
            logger.info("Skipping training set download")
            training_set = get_training_set_from_abc(mod_abbreviation=args.mod_train, topic=args.datatype_train,
                                                     metadata_only=True)
        else:
            training_set = get_training_set_from_abc(mod_abbreviation=args.mod_train, topic=args.datatype_train)
            reference_ids_positive = [agrkbid for agrkbid, positive in training_set["data_training"].items() if positive]
            reference_ids_negative = [agrkbid for agrkbid, positive in training_set["data_training"].items() if not positive]
            shutil.rmtree(os.path.join(training_data_dir, "positive"), ignore_errors=True)
            shutil.rmtree(os.path.join(training_data_dir, "negative"), ignore_errors=True)
            os.makedirs(os.path.join(training_data_dir, "positive"), exist_ok=True)
            os.makedirs(os.path.join(training_data_dir, "negative"), exist_ok=True)
            download_tei_files_from_abc_or_convert_pdf(reference_ids_positive, reference_ids_negative,
                                                       output_dir=training_data_dir,
                                                       mod_abbreviation=args.mod_train)
        classifier, stats = train_classifier(
            embedding_model_path=args.embedding_model_path,
            training_data_dir=training_data_dir,
            weighted_average_word_embedding=args.weighted_average_word_embedding,
            standardize_embeddings=args.standardize_embeddings, normalize_embeddings=args.normalize_embeddings,
            sections_to_use=args.sections_to_use)
        save_classifier(classifier=classifier, mod_abbreviation=args.mod_train, topic=args.datatype_train, stats=stats,
                        dataset_id=training_set["dataset_id"])
