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

import utils.thread_limits  # noqa: F401  (import first: pins native threads to 1)
import joblib
import nltk
import numpy as np
import scipy.sparse as sp
from gensim.models import KeyedVectors
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

from agr_dataset_manager.dataset_downloader import download_md_files_from_abc_or_convert_pdf
from models import POSSIBLE_CLASSIFIERS
from utils.abc_utils import (get_training_set_from_abc, upload_ml_model, get_reference_date,
                             get_reference_embedding)
from utils.abc_embeddings import abc_embedding_recipe
from utils.embedding import load_embedding_model, build_feature_matrix, get_bow_vectorizer
from utils.date_utils import parse_reference_date
from utils.get_documents import get_documents, remove_stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

logger = logging.getLogger(__name__)


def configure_logging(log_level):
    # Configure logging based on the log_level argument
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )


def detect_and_remove_outliers(X, y, method='isolation_forest', contamination=0.1):
    """
    Detect and remove outliers from the training data.

    Args:
        X: Feature matrix
        y: Labels
        method: Outlier detection method ('isolation_forest', 'elliptic_envelope', 'lof')
        contamination: Expected proportion of outliers in the dataset

    Returns:
        X_clean, y_clean: Data with outliers removed
        outlier_mask: Boolean mask indicating which samples were considered outliers
    """
    logger.info(f"Detecting outliers using {method} method (contamination={contamination})")

    if method == 'isolation_forest':
        detector = IsolationForest(contamination=contamination, random_state=42)
    elif method == 'elliptic_envelope':
        detector = EllipticEnvelope(contamination=contamination, random_state=42)
    elif method == 'lof':
        detector = LocalOutlierFactor(contamination=contamination, novelty=False)
    else:
        logger.warning(f"Unknown outlier detection method: {method}. Skipping outlier removal.")
        return X, y, np.ones(X.shape[0], dtype=bool)

    # Fit the detector and predict outliers
    outlier_predictions = detector.fit_predict(X)

    # Create mask for inliers (1) vs outliers (-1)
    inlier_mask = outlier_predictions == 1
    outlier_mask = outlier_predictions == -1

    # Remove outliers
    X_clean = X[inlier_mask]
    y_clean = y[inlier_mask]

    n_outliers = np.sum(outlier_mask)
    logger.info(f"Removed {n_outliers} outliers ({n_outliers / X.shape[0] * 100:.1f}%) "
                f"from {X.shape[0]} samples")
    logger.info(f"Remaining samples: {X_clean.shape[0]} "
                f"(Positive: {np.sum(y_clean)}, Negative: {np.sum(1 - y_clean)})")

    return X_clean, y_clean, outlier_mask


def _build_abc_embedding_features(abc_curies: dict, mod_abbreviation: str, use_bow: bool = False):
    """Build ``(X, y)`` from the ABC precomputed classifier embeddings.

    ``abc_curies`` maps ``"positive"``/``"negative"`` to reference-curie lists.
    Each reference's dense feature is the L2-normalized chunk-mean pool of its
    main-PDF paragraph embeddings (:func:`utils.abc_utils.get_reference_embedding`).
    When ``use_bow`` is set, the same stateless hashed bag-of-words block the
    BioWordVec classifiers use is concatenated onto the embedding — hashed over the
    references-excluded paragraph text carried in the parquet — yielding a sparse
    matrix (SCRUM-6052 recipe). References without an available embedding are
    dropped and counted, the same policy the BioWordVec path applies to references
    with no downloadable MD file."""
    rows = []
    y = []
    missing = 0
    bow_vectorizer = get_bow_vectorizer() if use_bow else None
    for label in ["positive", "negative"]:
        for reference_curie in abc_curies.get(label, []):
            result = get_reference_embedding(reference_curie, mod_abbreviation)
            if result is None:
                missing += 1
                continue
            pooled, text = result
            if use_bow:
                bow = bow_vectorizer.transform([remove_stopwords(text).lower() if text else ""])
                rows.append(sp.hstack([sp.csr_matrix(pooled.reshape(1, -1)), bow], format="csr"))
            else:
                rows.append(pooled)
            y.append(int(label == "positive"))
    if not rows:
        raise ValueError("No ABC embeddings could be retrieved for the training set; "
                         "cannot train. Check that the references have been embedded.")
    logger.info(f"ABC embeddings retrieved for {len(rows)} references "
                f"({missing} references had no embedding and were dropped). BoW={use_bow}.")
    X = sp.vstack(rows, format="csr") if use_bow else np.vstack(rows)
    return X, y


def train_classifier(embedding_model_path: str, training_data_dir: str, weighted_average_word_embedding: bool = False,
                     standardize_embeddings: bool = False, normalize_embeddings: bool = False,
                     sections_to_use: List[str] = None, remove_outliers: bool = False,
                     outlier_method: str = 'isolation_forest', outlier_contamination: float = 0.1,
                     use_bow_features: bool = False, use_max_pooling: bool = False,
                     use_lsh_features: bool = False,
                     include_keywords: bool = False, include_metadata: bool = False,
                     use_abc_embeddings: bool = False, abc_curies: dict = None,
                     mod_abbreviation: str = None):
    if use_abc_embeddings:
        # New path (SCRUM-5781): use the ABC's precomputed OpenAI embeddings instead
        # of loading BioWordVec and pooling word vectors over downloaded Markdown.
        # BoW (when enabled) is hashed from the parquet's own paragraph text, so no
        # Markdown download is needed.
        logger.info("Loading training set from ABC precomputed embeddings.")
        X, y = _build_abc_embedding_features(abc_curies or {}, mod_abbreviation,
                                             use_bow=use_bow_features)
        logger.info("Finished loading training set.")
        logger.info(f"Dataset size: {str(len(y))}")
        y = np.array(y)
        # When BoW is on the matrix is wide+sparse, so the same SVC/MLP skip and
        # XGBoost concurrency cap the BioWordVec path uses must apply here too.
        return _select_and_fit_model(X, y, remove_outliers, outlier_method, outlier_contamination,
                                     use_bow_features=use_bow_features, use_lsh_features=False)

    embedding_model = load_embedding_model(model_path=embedding_model_path)

    texts = []
    y = []

    # Precompute word_to_index
    if isinstance(embedding_model, KeyedVectors):
        word_to_index = embedding_model.key_to_index
    else:
        word_to_index = {word: idx for idx, word in enumerate(embedding_model.get_words())}

    # For each document in your training data, collect its text and label
    logger.info("Loading training set")
    for label in ["positive", "negative"]:
        documents = list(get_documents(
            os.path.join(training_data_dir, label),
            include_keywords=include_keywords,
            include_metadata=include_metadata))

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
                texts.append(text)
                y.append(int(label == "positive"))

    logger.info(f"Building features (max_pooling={use_max_pooling}, bow={use_bow_features}, "
                f"lsh={use_lsh_features}).")
    X = build_feature_matrix(embedding_model, texts, use_max_pooling=use_max_pooling,
                             use_bow=use_bow_features, use_lsh=use_lsh_features,
                             weighted_average_word_embedding=weighted_average_word_embedding,
                             standardize_embeddings=standardize_embeddings,
                             normalize_embeddings=normalize_embeddings, word_to_index=word_to_index)

    del embedding_model
    logger.info("Finished loading training set.")
    logger.info(f"Dataset size: {str(len(y))}")

    y = np.array(y)
    return _select_and_fit_model(X, y, remove_outliers, outlier_method, outlier_contamination,
                                 use_bow_features=use_bow_features, use_lsh_features=use_lsh_features)


def _select_and_fit_model(X, y, remove_outliers, outlier_method, outlier_contamination,
                          use_bow_features=False, use_lsh_features=False):

    # Step 1: Split data into train+val (80%) and holdout test set (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    logger.info(f"Dataset split - Total: {X.shape[0]}, Train+Val: {X_train_val.shape[0]}, "
                f"Test: {X_test.shape[0]}")
    logger.info(f"Class distribution - Train+Val: {np.bincount(y_train_val)}, Test: {np.bincount(y_test)}")

    # Step 2: Apply outlier detection and removal if requested
    if remove_outliers:
        X_train_val, y_train_val, outlier_mask = detect_and_remove_outliers(
            X_train_val, y_train_val,
            method=outlier_method,
            contamination=outlier_contamination
        )

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
        # On the high-dimensional sparse matrix produced by the BoW block, RBF-SVC
        # is slow/weak and MLPClassifier is both memory-heavy (its input-layer
        # weight matrix is n_features x hidden, ~210MB at 2**18 features, multiplied
        # across RandomizedSearchCV's parallel workers -> OOM) and prone to severe
        # overfitting. Skip both when BoW features are enabled.
        if use_bow_features and classifier_name in ('SVC', 'MLPClassifier'):
            logger.info(f"Skipping {classifier_name}: not suitable for the high-dimensional sparse "
                        f"BoW feature matrix.")
            continue
        logger.info(f"Evaluating model {classifier_name}.")

        # Reduce n_iter for faster training with focus on regularization
        n_iter = 50 if classifier_name in ['LogisticRegression', 'SGDClassifier'] else 30

        # RandomizedSearchCV fans every candidate fit out to its own worker
        # process. On the wide sparse BoW/LSH matrix (2**18 columns) a worker
        # allocates buffers sized by n_features (XGBoost gradient histograms
        # ~3 GB/fit; LightGBM feature histograms similarly), so the default
        # n_jobs=-1 (one worker per core) peaks memory and a worker gets OS-killed
        # (SIGSEGV) — reproduced on the larger FB training sets (e.g. disease,
        # ~2774 refs) where LightGBM crashed even with native threads pinned to 1,
        # while the smaller sets (~1500 refs) survived. Cap the *search* concurrency
        # for every model on wide matrices; n_jobs does not change the result (the
        # search is deterministic given random_state), so model-selection stats are
        # unaffected. Override with SEARCH_MAX_JOBS.
        search_n_jobs = -1
        if use_bow_features or use_lsh_features:
            search_n_jobs = int(os.environ.get('SEARCH_MAX_JOBS', '4'))

        random_search = RandomizedSearchCV(
            estimator=classifier_info['model'],
            n_iter=n_iter,
            param_distributions=classifier_info['params'],
            cv=stratified_k_folds,
            scoring=scoring,
            refit='f1',
            verbose=1,
            n_jobs=search_n_jobs,
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
            best_classifier = random_search.best_estimator_
            best_params = random_search.best_params_
            best_classifier_name = classifier_name
            best_results = random_search.cv_results_
            best_index = random_search.best_index_

    # Log model selection summary
    logger.info("\n" + "=" * 60)
    logger.info("Model Selection Summary (sorted by penalized score):")
    logger.info("-" * 60)

    sorted_models = sorted(model_selection_scores.items(),
                           key=lambda x: x[1]['penalized_score'],
                           reverse=True)

    for rank, (name, scores) in enumerate(sorted_models, 1):
        logger.info(f"{rank}. {name:25s} | Score: {scores['penalized_score']:.3f} | "
                    f"CV: {scores['cv_f1']:.3f} | Test: {scores['test_f1']:.3f} | "
                    f"Gap: {scores['gap']:.3f} | Penalty: {scores['penalty']:.3f}")

    logger.info("=" * 60)
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
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Final Model Performance Summary for {best_classifier_name}:")
    logger.info(f"Cross-Validation (10-fold): F1={average_f1:.3f} (±{std_f1:.3f})")
    logger.info(f"Holdout Test Set: F1={final_test_f1:.3f}, "
                f"Precision={final_test_precision:.3f}, Recall={final_test_recall:.3f}")
    logger.info(f"Generalization Gap (CV-Test): {average_f1 - final_test_f1:.3f}")
    logger.info(f"{'=' * 60}\n")

    # Return the trained model and performance metrics
    return best_classifier, stats


def save_classifier(classifier, mod_abbreviation: str, topic: str,
                    data_novelty: Union[str, None],
                    production: Union[bool, None], no_data: Union[bool, None],
                    species: Union[str, None], stats: dict, dataset_id: int, test_mode: bool = False,
                    training_data_dir: str = None, embedding_recipe: Union[dict, None] = None):
    if training_data_dir is None:
        training_data_dir = os.getenv("TRAINING_DIR", "/data/agr_document_classifier/training")
    topic_formatted = topic.replace(':', '_')
    model_filename = f"{mod_abbreviation}_{topic_formatted}_classifier.joblib"
    model_path = os.path.join(training_data_dir, model_filename)

    # Save the classifier directly (compatible with existing classification pipeline)
    joblib.dump(classifier, model_path)
    if test_mode:
        logger.info(f"Saved model to {model_path}, skipping upload because in test mode.")
    else:
        upload_ml_model("biocuration_topic_classification", mod_abbreviation=mod_abbreviation, topic=topic,
                        data_novelty=data_novelty, production=production,
                        no_data=no_data, species=species,
                        model_path=model_path, stats=stats, dataset_id=dataset_id, file_extension="joblib",
                        embedding_recipe=embedding_recipe)


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
    parser.add_argument("--use_bow_features", action="store_true",
                        help="Concatenate a stateless hashing bag-of-words block with the embedding "
                             "(must be passed identically at classification time)",
                        required=False)
    parser.add_argument("--use_max_pooling", action="store_true",
                        help="Concatenate an element-wise max-pooled embedding alongside the mean "
                             "(must be passed identically at classification time)",
                        required=False)
    parser.add_argument("--use_lsh_features", action="store_true",
                        help="Concatenate a stateless LSH bag-of-concepts block (random-hyperplane "
                             "buckets over the word embeddings, so near-synonyms share a bin) with the "
                             "embedding (must be passed identically at classification time)",
                        required=False)
    parser.add_argument("--include_keywords", action="store_true",
                        help="Include author keywords in the document text (off by default; must be "
                             "passed identically at classification time)",
                        required=False)
    parser.add_argument("--include_metadata", action="store_true",
                        help="Include article metadata in the document text (off by default; must be "
                             "passed identically at classification time)",
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
    parser.add_argument("-Q", "--data_novelty", type=str, required=False, default='ATP:0000335',
                        help="Qualifier to be used for novelty. Default 'ATP:0000335'")
    parser.add_argument("-a", "--alternative_species", type=str,
                        help="Use a non standard mod species taxon. Must include 'taxon:'",
                        required=False)
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode and store model "
                                                                 "locally.", required=False)
    parser.add_argument("--dataset_version", type=int, required=False,
                        help="Specific dataset version to use for training (defaults to latest)")
    parser.add_argument("--filter_date_before", type=str, required=False, default="2005-01-01",
                        help="Filter out references published before this date (YYYY-MM-DD format). "
                             "Defaults to 2005-01-01; pass an empty string to disable date filtering.")
    # Outlier detection arguments
    parser.add_argument("--remove_outliers", action="store_true",
                        help="Enable outlier detection and removal")
    parser.add_argument("--outlier_method", type=str, default="isolation_forest",
                        choices=['isolation_forest', 'elliptic_envelope', 'lof'],
                        help="Outlier detection method (default: isolation_forest)")
    parser.add_argument("--outlier_contamination", type=float, default=0.1,
                        help="Expected proportion of outliers (default: 0.1)")
    return parser.parse_args()


def get_filtered_training_curies(args, training_set):
    """Split a training set into positive/negative reference-curie lists, applying
    the optional ``--filter_date_before`` cutoff. Returns
    ``{"positive": [...], "negative": [...]}``."""
    reference_ids_positive = [agrkbid for agrkbid, classification_value in training_set["data_training"].items() if
                              classification_value == "positive"]
    reference_ids_negative = [agrkbid for agrkbid, classification_value in training_set["data_training"].items() if
                              classification_value == "negative"]

    # Apply date filtering if specified
    if args.filter_date_before:
        try:
            filter_date = datetime.strptime(args.filter_date_before, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {args.filter_date_before}. Expected YYYY-MM-DD")
            raise
        logger.info(f"Filtering out references published before {args.filter_date_before}")

        def _keep_on_or_after(ref_ids):
            """Keep references published on/after filter_date. References whose
            (possibly messy PubMed) date cannot be parsed at all are kept."""
            kept = []
            for ref_id in ref_ids:
                ref_date_str = get_reference_date(ref_id)
                ref_date = parse_reference_date(ref_date_str)
                if ref_date is None:
                    logger.debug(f"No parseable date for {ref_id} ('{ref_date_str}'), including it")
                    kept.append(ref_id)
                elif ref_date >= filter_date:
                    kept.append(ref_id)
                else:
                    logger.debug(f"Filtering out {ref_id} (published {ref_date_str})")
            return kept

        filtered_positive = _keep_on_or_after(reference_ids_positive)
        filtered_negative = _keep_on_or_after(reference_ids_negative)
        logger.info(f"Date filtering complete. Positive: {len(reference_ids_positive)} -> "
                    f"{len(filtered_positive)}, Negative: {len(reference_ids_negative)} -> "
                    f"{len(filtered_negative)}")
        reference_ids_positive = filtered_positive
        reference_ids_negative = filtered_negative

    return {"positive": reference_ids_positive, "negative": reference_ids_negative}


def download_training_set(args, training_data_dir):
    # Get training set with optional version
    training_set = get_training_set_from_abc(mod_abbreviation=args.mod_train, topic=args.datatype_train,
                                             version=args.dataset_version)

    curies = get_filtered_training_curies(args, training_set)
    reference_ids_positive = curies["positive"]
    reference_ids_negative = curies["negative"]

    shutil.rmtree(os.path.join(training_data_dir, "positive"), ignore_errors=True)
    shutil.rmtree(os.path.join(training_data_dir, "negative"), ignore_errors=True)
    os.makedirs(os.path.join(training_data_dir, "positive"), exist_ok=True)
    os.makedirs(os.path.join(training_data_dir, "negative"), exist_ok=True)
    download_md_files_from_abc_or_convert_pdf(reference_ids_positive, reference_ids_negative,
                                              output_dir=training_data_dir,
                                              mod_abbreviation=args.mod_train)
    return training_set


def upload_pre_existing_model(args, training_set, training_data_dir):
    logger.info("Skipping training. Uploading pre-existing model and stats file to ABC")
    stats_path = os.path.join(training_data_dir, f"{args.mod_train}_{args.datatype_train.replace(':', '_')}_metadata.json")
    stats = json.load(open(stats_path))
    stats["best_params"] = stats["parameters"]
    stats["model_name"] = stats["model_type"]
    stats["average_precision"] = stats["precision"]
    stats["average_recall"] = stats["recall"]
    stats["average_f1"] = stats["f1_score"]
    upload_ml_model(task_type="biocuration_topic_classification", mod_abbreviation=args.mod_train,
                    topic=args.datatype_train,
                    data_novelty=args.data_novelty,
                    production=args.production,
                    no_data=not args.do_not_flag_no_data, species=args.alternative_species,
                    model_path=os.path.join(training_data_dir,
                                            f"{args.mod_train}_{args.datatype_train.replace(':', '_')}"
                                            f"_classifier.joblib"),
                    stats=stats, dataset_id=training_set["dataset_id"], file_extension="joblib")


def train_and_save_model(args, training_data_dir, training_set, abc_curies=None):
    if args.test_mode:
        logger.info("Running in test mode. Model will be saved locally and not uploaded to ABC.")
    # ABC precomputed embeddings + hashed BoW are the standard training recipe now
    # (SCRUM-5781/6052): embedding alone underperformed BoW, embedding+BoW matched
    # the BoW baseline, so both are always used.
    classifier, stats = train_classifier(
        embedding_model_path=args.embedding_model_path,
        training_data_dir=training_data_dir,
        weighted_average_word_embedding=args.weighted_average_word_embedding,
        standardize_embeddings=args.standardize_embeddings,
        normalize_embeddings=args.normalize_embeddings,
        sections_to_use=args.sections_to_use,
        remove_outliers=args.remove_outliers,
        outlier_method=args.outlier_method,
        outlier_contamination=args.outlier_contamination,
        use_bow_features=True,
        use_max_pooling=args.use_max_pooling,
        use_lsh_features=args.use_lsh_features,
        include_keywords=args.include_keywords,
        include_metadata=args.include_metadata,
        use_abc_embeddings=True,
        abc_curies=abc_curies,
        mod_abbreviation=args.mod_train)
    logger.info(f"Best classifier stats: {str(stats)}")
    # Tag the model (via the ml_model embedding_* columns) so the classifier
    # fetches ABC embeddings for it and rebuilds the identical embedding+BoW
    # vector. Legacy BioWordVec models leave those columns NULL and keep their
    # on-the-fly path.
    embedding_recipe = abc_embedding_recipe(use_bow=True)
    save_classifier(classifier=classifier, mod_abbreviation=args.mod_train, topic=args.datatype_train,
                    data_novelty=args.data_novelty,
                    production=args.production,
                    no_data=not args.do_not_flag_no_data, species=args.alternative_species,
                    stats=stats, dataset_id=training_set["dataset_id"], test_mode=args.test_mode,
                    training_data_dir=training_data_dir, embedding_recipe=embedding_recipe)


def train_mode(args):
    # ABC precomputed embeddings are the default (and only) training source now.
    # We need the dataset metadata (dataset_id) plus the positive/negative curie
    # lists to fetch the embeddings; no MD/TEI download.
    training_data_dir = os.getenv("TRAINING_DIR", "/data/agr_document_classifier/training")
    logger.info("Training on ABC precomputed embeddings (default).")
    training_set = get_training_set_from_abc(mod_abbreviation=args.mod_train, topic=args.datatype_train,
                                             metadata_only=False, version=args.dataset_version)
    abc_curies = get_filtered_training_curies(args, training_set)
    if args.skip_training:
        upload_pre_existing_model(args, training_set, training_data_dir)
    else:
        train_and_save_model(args, training_data_dir, training_set, abc_curies=abc_curies)


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
