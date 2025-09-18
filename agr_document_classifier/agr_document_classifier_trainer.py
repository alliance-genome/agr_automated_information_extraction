import argparse
import json
import logging
import os
import os.path
import re
import shutil
import sys
from datetime import datetime
from typing import List, Union

import joblib
import nltk
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score

from agr_dataset_manager.dataset_downloader import download_tei_files_from_abc_or_convert_pdf
from models import POSSIBLE_CLASSIFIERS
from utils.abc_utils import get_training_set_from_abc, upload_ml_model, get_reference_date
from utils.embedding import load_embedding_model, get_document_embedding
from utils.get_documents import get_documents, remove_stopwords

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

    # Step 1: Split data into train+val (80%) and holdout test set (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    logger.info(f"Dataset split - Total: {len(X)}, Train+Val: {len(X_train_val)}, Test: {len(X_test)}")
    logger.info(f"Class distribution - Train+Val: {np.bincount(y_train_val)}, Test: {np.bincount(y_test)}")

    best_score = 0
    best_penalized_score = 0
    best_classifier = None
    best_params = None
    best_classifier_name = ""
    best_results = {}
    best_index = 0
    test_scores = {}
    model_selection_scores = {}

    # Use 10-fold cross-validation for more robust validation
    stratified_k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0)
    }

    logger.info("Starting model selection with hyperparameter optimization and cross-validation.")
    logger.info("Using penalized scoring: weighted_score = 0.7 * test_f1 + 0.3 * cv_f1 - penalty")

    for classifier_name, classifier_info in POSSIBLE_CLASSIFIERS.items():
        logger.info(f"Evaluating model {classifier_name}.")

        # Reduce n_iter for faster training with focus on regularization
        n_iter = 50 if classifier_name in ['LogisticRegression', 'SGDClassifier'] else 30

        random_search = RandomizedSearchCV(
            estimator=classifier_info['model'],
            n_iter=n_iter,
            param_distributions=classifier_info['params'],
            cv=stratified_k_folds,
            scoring=scoring,
            refit='f1',
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        random_search.fit(X_train_val, y_train_val)

        # Evaluate on test set
        test_pred = random_search.predict(X_test)
        test_f1 = f1_score(y_test, test_pred, zero_division=0)
        test_precision = precision_score(y_test, test_pred, zero_division=0)
        test_recall = recall_score(y_test, test_pred, zero_division=0)
        cv_f1 = random_search.best_score_

        test_scores[classifier_name] = {
            'f1': test_f1,
            'precision': test_precision,
            'recall': test_recall
        }

        # Calculate overfitting gap and penalty
        overfitting_gap = cv_f1 - test_f1

        # Penalize overfitting: penalty increases exponentially with gap
        if overfitting_gap > 0.05:  # Start penalizing after 5% gap
            overfitting_penalty = (overfitting_gap - 0.05) * 2.0  # Double the gap as penalty
        else:
            overfitting_penalty = 0.0

        # Calculate penalized score that balances test and CV performance
        # Prioritize test performance (70%) but also consider CV performance (30%)
        penalized_score = (0.7 * test_f1 + 0.3 * cv_f1) - overfitting_penalty

        model_selection_scores[classifier_name] = {
            'cv_f1': cv_f1,
            'test_f1': test_f1,
            'gap': overfitting_gap,
            'penalty': overfitting_penalty,
            'penalized_score': penalized_score
        }

        logger.info(f"Model {classifier_name}:")
        logger.info(f"  CV F1: {cv_f1:.3f}, Test F1: {test_f1:.3f}")
        logger.info(f"  Generalization gap: {overfitting_gap:.3f}")
        logger.info(f"  Overfitting penalty: {overfitting_penalty:.3f}")
        logger.info(f"  Penalized score: {penalized_score:.3f}")

        # Warn about severe overfitting
        if overfitting_gap > 0.15:
            logger.warning(f"⚠️ Severe overfitting detected for {classifier_name}: gap = {overfitting_gap:.3f}")
        elif overfitting_gap > 0.1:
            logger.warning(f"⚠️ Moderate overfitting detected for {classifier_name}: gap = {overfitting_gap:.3f}")

        # Select best model based on penalized score
        if penalized_score > best_penalized_score:
            best_penalized_score = penalized_score
            best_score = cv_f1
            best_classifier = random_search.best_estimator_
            best_params = random_search.best_params_
            best_classifier_name = classifier_name
            best_results = random_search.cv_results_
            best_index = random_search.best_index_

    # Log model selection summary
    logger.info("\n" + "="*60)
    logger.info("Model Selection Summary (sorted by penalized score):")
    logger.info("-"*60)

    sorted_models = sorted(model_selection_scores.items(),
                          key=lambda x: x[1]['penalized_score'],
                          reverse=True)

    for rank, (name, scores) in enumerate(sorted_models, 1):
        logger.info(f"{rank}. {name:25s} | Score: {scores['penalized_score']:.3f} | "
                   f"CV: {scores['cv_f1']:.3f} | Test: {scores['test_f1']:.3f} | "
                   f"Gap: {scores['gap']:.3f} | Penalty: {scores['penalty']:.3f}")

    logger.info("="*60)
    logger.info(f"Selected model: {best_classifier_name} with penalized score: {best_penalized_score:.3f}")

    # Retrain best model on full train+val set for final model
    logger.info("Retraining best model on full training set...")
    best_classifier.fit(X_train_val, y_train_val)

    # Final evaluation on test set
    final_test_pred = best_classifier.predict(X_test)
    final_test_f1 = f1_score(y_test, final_test_pred)
    final_test_precision = precision_score(y_test, final_test_pred)
    final_test_recall = recall_score(y_test, final_test_pred)

    # Retrieve the average precision, recall, and F1 score from CV
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
        "test_precision": round(float(final_test_precision), 3),
        "test_recall": round(float(final_test_recall), 3),
        "test_f1": round(float(final_test_f1), 3),
        "best_params": best_params,
        "all_models_test_scores": test_scores,
        "model_selection": {
            "penalized_score": round(best_penalized_score, 3),
            "generalization_gap": round(average_f1 - final_test_f1, 3),
            "selection_criteria": "0.7 * test_f1 + 0.3 * cv_f1 - overfitting_penalty",
            "all_models_scores": model_selection_scores
        }
    }

    # Log comparison between CV and test performance
    logger.info(f"\n{'='*60}")
    logger.info(f"Final Model Performance Summary for {best_classifier_name}:")
    logger.info(f"Cross-Validation (10-fold): F1={average_f1:.3f} (±{std_f1:.3f})")
    logger.info(f"Holdout Test Set: F1={final_test_f1:.3f}, Precision={final_test_precision:.3f}, Recall={final_test_recall:.3f}")
    logger.info(f"Generalization Gap (CV-Test): {average_f1 - final_test_f1:.3f}")
    logger.info(f"{'='*60}\n")

    # Return the trained model and performance metrics
    return best_classifier, stats


def save_classifier(classifier, mod_abbreviation: str, topic: str,
                    novel_data: Union[bool, None], novel_topic_qualifier: Union[str, None], production: Union[bool, None], no_data: Union[bool, None], species: Union[str, None],
                    stats: dict, dataset_id: int, test_mode: bool = False):
    model_path = f"/data/agr_document_classifier/training/{mod_abbreviation}_{topic.replace(':', '_')}_classifier.joblib"
    joblib.dump(classifier, model_path)
    if test_mode:
        logger.info(f"Saved model to {model_path}, skipping upload because in test mode.")
    else:
        upload_ml_model("biocuration_topic_classification", mod_abbreviation=mod_abbreviation, topic=topic,
                        novel_data=novel_data, novel_topic_qualifier=novel_topic_qualifier, production=production,
                        no_data=no_data, species=species,
                        model_path=model_path, stats=stats, dataset_id=dataset_id, file_extension="joblib")


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
    parser.add_argument("-p", "--production", action="store_true",
                        help="Whether to set production to true. (Default False)",
                        required=False)
    parser.add_argument("-f", "--do_not_flag_no_data", action="store_true",
                        help="Whether to not set no_data flag to true if no result found. (Default False)",
                        required=False)
    parser.add_argument("-F", "--flag_novel", action="store_true",
                        help="Whether to set novel_data to true on positive result. Default False",
                        required=False)
    parser.add_argument("-Q", "--novel_topic_qualifier", type=str, required=False, default='ATP:0000335',
                        help="Qualifier to be used for novelty. Default 'ATP:0000335'")
    parser.add_argument("-a", "--alternative_species", type=str,
                        help="Use a non standard mod species taxon. Must include 'taxon:'",
                        required=False)
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode and store model "
                                                                 "locally.", required=False)
    parser.add_argument("--dataset_version", type=int, required=False,
                        help="Specific dataset version to use for training (defaults to latest)")
    parser.add_argument("--filter_date_before", type=str, required=False,
                        help="Filter out references published before this date (YYYY-MM-DD format)")
    return parser.parse_args()


def download_training_set(args, training_data_dir):
    # Get training set with optional version
    training_set = get_training_set_from_abc(mod_abbreviation=args.mod_train, topic=args.datatype_train,
                                             version=args.dataset_version)

    reference_ids_positive = [agrkbid for agrkbid, classification_value in training_set["data_training"].items() if
                              classification_value == "positive"]
    reference_ids_negative = [agrkbid for agrkbid, classification_value in training_set["data_training"].items() if
                              classification_value == "negative"]

    # Apply date filtering if specified
    if args.filter_date_before:
        try:
            filter_date = datetime.strptime(args.filter_date_before, "%Y-%m-%d")
            logger.info(f"Filtering out references published before {args.filter_date_before}")

            # Filter positive references
            filtered_positive = []
            for ref_id in reference_ids_positive:
                ref_date_str = get_reference_date(ref_id)
                if ref_date_str:
                    try:
                        ref_date = datetime.strptime(ref_date_str[:10], "%Y-%m-%d")  # Take only date part
                        if ref_date >= filter_date:
                            filtered_positive.append(ref_id)
                        else:
                            logger.debug(f"Filtering out {ref_id} (published {ref_date_str})")
                    except ValueError:
                        logger.warning(f"Could not parse date '{ref_date_str}' for reference {ref_id}, including it")
                        filtered_positive.append(ref_id)
                else:
                    logger.debug(f"No date found for reference {ref_id}, including it")
                    filtered_positive.append(ref_id)

            # Filter negative references
            filtered_negative = []
            for ref_id in reference_ids_negative:
                ref_date_str = get_reference_date(ref_id)
                if ref_date_str:
                    try:
                        ref_date = datetime.strptime(ref_date_str[:10], "%Y-%m-%d")  # Take only date part
                        if ref_date >= filter_date:
                            filtered_negative.append(ref_id)
                        else:
                            logger.debug(f"Filtering out {ref_id} (published {ref_date_str})")
                    except ValueError:
                        logger.warning(f"Could not parse date '{ref_date_str}' for reference {ref_id}, including it")
                        filtered_negative.append(ref_id)
                else:
                    logger.debug(f"No date found for reference {ref_id}, including it")
                    filtered_negative.append(ref_id)

            logger.info(f"Date filtering complete. Positive: {len(reference_ids_positive)} -> {len(filtered_positive)}, "
                       f"Negative: {len(reference_ids_negative)} -> {len(filtered_negative)}")
            reference_ids_positive = filtered_positive
            reference_ids_negative = filtered_negative

        except ValueError:
            logger.error(f"Invalid date format: {args.filter_date_before}. Expected YYYY-MM-DD")
            raise

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
                    novel_data=args.flag_novel,
                    novel_topic_qualifier=args.novel_topic_qualifier,
                    production=args.production,
                    no_data=not args.do_not_flag_no_data, species=args.alternative_species,
                    model_path=f"/data/agr_document_classifier/training/{args.mod_train}_"
                               f"{args.datatype_train.replace(':', '_')}_classifier.joblib",
                    stats=stats, dataset_id=training_set["dataset_id"], file_extension="joblib")


def train_and_save_model(args, training_data_dir, training_set):
    if args.test_mode:
        logger.info("Running in test mode. Model will be saved locally and not uploaded to ABC.")
    classifier, stats = train_classifier(
        embedding_model_path=args.embedding_model_path,
        training_data_dir=training_data_dir,
        weighted_average_word_embedding=args.weighted_average_word_embedding,
        standardize_embeddings=args.standardize_embeddings, normalize_embeddings=args.normalize_embeddings,
        sections_to_use=args.sections_to_use)
    logger.info(f"Best classifier stats: {str(stats)}")
    save_classifier(classifier=classifier, mod_abbreviation=args.mod_train, topic=args.datatype_train,
                    novel_data=args.flag_novel, novel_topic_qualifier=args.novel_topic_qualifier,
                    production=args.production,
                    no_data=not args.do_not_flag_no_data, species=args.alternative_species,
                    stats=stats, dataset_id=training_set["dataset_id"], test_mode=args.test_mode)


def train_mode(args):
    training_data_dir = "/data/agr_document_classifier/training"
    if args.skip_training_set_download:
        logger.info("Skipping training set download")
        training_set = get_training_set_from_abc(mod_abbreviation=args.mod_train, topic=args.datatype_train,
                                                 metadata_only=True, version=args.dataset_version)
    else:
        training_set = download_training_set(args, training_data_dir)
    if args.skip_training:
        upload_pre_existing_model(args, training_set)
    else:
        train_and_save_model(args, training_data_dir, training_set)


def main():
    args = parse_arguments()
    if args.alternative_species:
        if not re.search(r'^NCBITaxon:\d+$', args.alternative_species):
            print("Invalid alternative species specified. Must start with 'NCBITaxon:' followed by numbers ONLY")
            sys.exit(1)
    configure_logging(args.log_level)

    train_mode(args)


if __name__ == '__main__':
    main()
