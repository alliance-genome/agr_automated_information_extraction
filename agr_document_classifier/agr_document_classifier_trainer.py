import argparse
import json
import logging
import os
import os.path
import shutil
import sys
from typing import List

import joblib
import nltk
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from agr_dataset_manager.dataset_downloader import download_tei_files_from_abc_or_convert_pdf
from models import POSSIBLE_CLASSIFIERS
from utils.abc_utils import get_training_set_from_abc, upload_ml_model
from utils.embedding import load_embedding_model, get_document_embedding
from utils.get_documents import get_documents

nltk.download('stopwords')
nltk.download('punkt')

logger = logging.getLogger(__name__)


def configure_logging(log_level):
    # Configure logging based on the log_level argument
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )


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
        documents = list(get_documents(os.path.join(training_data_dir, label)))

        for _, (_, fulltext, title, abstract) in enumerate(documents, start=1):
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
    upload_ml_model("biocuration_topic_classification", mod_abbreviation=mod_abbreviation, topic=topic,
                    model_path=model_path, stats=stats, dataset_id=dataset_id, file_extension="joblib")


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)


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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train document classifiers')
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
    parser.add_argument("-N", "--skip_training", action="store_true",
                        help="Just upload a pre-existing model and stats file to the ABC without training",
                        required=False)
    parser.add_argument("-l", "--log_level", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="Set the logging level")
    return parser.parse_args()


def download_training_set(args, training_data_dir):
    training_set = get_training_set_from_abc(mod_abbreviation=args.mod_train, topic=args.datatype_train)
    reference_ids_positive = [agrkbid for agrkbid, classification_value in training_set["data_training"].items() if
                              classification_value == "positive"]
    reference_ids_negative = [agrkbid for agrkbid, classification_value in training_set["data_training"].items() if
                              classification_value == "negative"]
    shutil.rmtree(os.path.join(training_data_dir, "positive"), ignore_errors=True)
    shutil.rmtree(os.path.join(training_data_dir, "negative"), ignore_errors=True)
    os.makedirs(os.path.join(training_data_dir, "positive"), exist_ok=True)
    os.makedirs(os.path.join(training_data_dir, "negative"), exist_ok=True)
    download_tei_files_from_abc_or_convert_pdf(reference_ids_positive, reference_ids_negative,
                                               output_dir=training_data_dir,
                                               mod_abbreviation=args.mod_train)
    return training_set


def upload_pre_existing_model(args, training_set):
    logger.info("Skipping training. Uploading pre-existing model and stats file to ABC")
    stats = json.load(open(f"/data/agr_document_classifier/training/{args.mod_train}_"
                           + f"{args.datatype_train.replace(':', '_')}_metadata.json"))
    stats["best_params"] = stats["parameters"]
    stats["model_name"] = stats["model_type"]
    stats["average_precision"] = stats["precision"]
    stats["average_recall"] = stats["recall"]
    stats["average_f1"] = stats["f1_score"]
    upload_ml_model(task_type="biocuration_topic_classification", mod_abbreviation=args.mod_train,
                    topic=args.datatype_train,
                    model_path=f"/data/agr_document_classifier/training/{args.mod_train}_"
                               f"{args.datatype_train.replace(':', '_')}_classifier.joblib",
                    stats=stats, dataset_id=training_set["dataset_id"], file_extension="joblib")


def train_and_save_model(args, training_data_dir, training_set):
    classifier, stats = train_classifier(
        embedding_model_path=args.embedding_model_path,
        training_data_dir=training_data_dir,
        weighted_average_word_embedding=args.weighted_average_word_embedding,
        standardize_embeddings=args.standardize_embeddings, normalize_embeddings=args.normalize_embeddings,
        sections_to_use=args.sections_to_use)
    logger.info(f"Best classifier stats: {str(stats)}")
    save_classifier(classifier=classifier, mod_abbreviation=args.mod_train, topic=args.datatype_train,
                    stats=stats, dataset_id=training_set["dataset_id"])


def train_mode(args):
    training_data_dir = "/data/agr_document_classifier/training"
    if args.skip_training_set_download:
        logger.info("Skipping training set download")
        training_set = get_training_set_from_abc(mod_abbreviation=args.mod_train, topic=args.datatype_train,
                                                 metadata_only=True)
    else:
        training_set = download_training_set(args, training_data_dir)
    if args.skip_training:
        upload_pre_existing_model(args, training_set)
    else:
        train_and_save_model(args, training_data_dir, training_set)


def main():
    args = parse_arguments()
    configure_logging(args.log_level)

    train_mode(args)


if __name__ == '__main__':
    main()
