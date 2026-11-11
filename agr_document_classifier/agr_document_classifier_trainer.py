import argparse
import inspect
import json
import logging
import os
import os.path
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional, Union

import utils.thread_limits  # noqa: F401  (import first: pins native threads to 1)
import joblib
import numpy as np
import scipy.sparse as sp
from gensim.models import KeyedVectors
from sklearn.base import clone
from sklearn.metrics import average_precision_score, make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

from agr_dataset_manager.dataset_downloader import download_md_files_from_abc_or_convert_pdf
from agr_document_classifier.models import POSSIBLE_CLASSIFIERS
from agr_document_classifier.rebalance import (POSITIVE_MIX_MODES, REBALANCE_MODES, Strata, balance_classes,
                                               bucket_counts, corpus_weights, format_counts,
                                               load_bucket_prevalence, load_strata, prevalence_weights,
                                               scored_buckets, subsample)
from utils.abc_utils import (get_training_set_from_abc, upload_ml_model, get_reference_metadata,
                             get_reference_embedding, get_reference_abstract_text,
                             default_data_context_for_mod)
from utils.abc_embeddings import (abc_embedding_recipe, get_profile, get_profile_by_name,
                                  registered_profile_names, ABC_EMBEDDING_PROFILE,
                                  ABC_EMBEDDING_VERSION, TEXT_SOURCE_REFERENCE_ABSTRACT)
from utils.embedding import load_embedding_model, build_feature_matrix, get_bow_vectorizer
from utils.date_utils import parse_reference_date
from utils.get_documents import get_documents, remove_stopwords

# No nltk.download here: importing utils.get_documents above already fetches
# stopwords/punkt/punkt_tab at import time and reports any failure.
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


def _cache_path(cache_dir: str, reference_curie: str, profile_name: str, version: int) -> str:
    return os.path.join(cache_dir, f"{profile_name}_v{version}", reference_curie.replace(":", "_") + ".npz")


def _read_cached_features(path: str):
    """``(pooled_or_None, text)`` from a cache file, or None if it is missing or unreadable."""
    try:
        with np.load(path, allow_pickle=False) as data:
            pooled = data["pooled"]
            return (pooled if pooled.size else None), str(data["text"])
    except (OSError, KeyError, ValueError):
        return None


def _write_cached_features(path: str, result) -> None:
    pooled, text = result
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp_path, pooled=np.zeros(0, dtype=np.float32) if pooled is None else pooled,
                        text=np.array(text or ""))
    os.replace(tmp_path, path)


def _fetch_reference_features(reference_curie: str, mod_abbreviation: str, profile_name: str, version: int,
                              text_from_reference: bool, cache_dir: Optional[str] = None):
    """``(pooled_or_None, text)`` for one reference, or None when unavailable.

    With ``cache_dir`` (SCRUM-6482), a reference fetched once is stored there, keyed by profile name and
    version, and later runs read it from disk instead of downloading the parquet again. Unavailable
    references are not cached, so they are retried on the next run. A re-embedding under the same profile
    version is not noticed: delete the cache directory to refresh it."""
    path = _cache_path(cache_dir, reference_curie, profile_name, version) if cache_dir else None
    if path and os.path.exists(path):
        cached = _read_cached_features(path)
        if cached is not None:
            return cached
    if text_from_reference:
        # BoW-only profile: no parquet, the reference record is the source.
        text_only = get_reference_abstract_text(reference_curie)
        result = None if text_only is None else (None, text_only)
    else:
        result = get_reference_embedding(reference_curie, mod_abbreviation,
                                         profile_name=profile_name, version=version)
    if result is not None and path:
        _write_cached_features(path, result)
    return result


def _build_abc_embedding_features(abc_curies: dict, mod_abbreviation: str, use_bow: bool = False,
                                  profile_name: str = ABC_EMBEDDING_PROFILE,
                                  version: int = ABC_EMBEDDING_VERSION,
                                  kept_curies: Optional[list] = None,
                                  cache_dir: Optional[str] = None, workers: int = 1):
    """Build ``(X, y)`` for the requested feature profile.

    ``abc_curies`` maps ``"positive"``/``"negative"`` to reference-curie lists. The
    profile decides the layout, and ``classify_documents_from_abc_embeddings``
    resolves it identically so train and classify cannot drift:

    * An *embedding* profile contributes the L2-normalized chunk-mean pool of its
      parquet's chunk embeddings, plus (when ``use_bow``) the stateless hashed
      bag-of-words block over the parquet's own text — the SCRUM-6052 recipe.
    * A *BoW-only* profile has no parquet: the text comes from the reference record
      (:func:`utils.abc_utils.get_reference_abstract_text`) and the row is the BoW
      block alone. This is what the ZFIN molecular-probe model ships on (SCRUM-5764),
      where the embedding block measured no better than BoW by itself.

    References whose features are unavailable — no embedding, or no title/abstract —
    are dropped and counted, the same policy the BioWordVec path applies to
    references with no downloadable MD file.

    When ``kept_curies`` is given, the curie of every row of ``X`` is appended to
    it, in row order (SCRUM-6482: per-row NNC bucket and weight).

    ``cache_dir`` keeps each fetched reference on disk for later runs and ``workers`` fetches that many
    references in parallel (SCRUM-6482); the row order does not depend on either."""
    rows = []
    y = []
    missing = 0
    bow_vectorizer = get_bow_vectorizer() if use_bow else None
    # The profile decides the feature layout; classify resolves it the same way, so
    # the two cannot drift. An unregistered profile keeps the fulltext behaviour.
    profile = get_profile(profile_name, version)
    use_embedding_block = profile.use_embedding if profile else True
    text_from_reference = (profile is not None
                           and profile.text_source == TEXT_SOURCE_REFERENCE_ABSTRACT)
    labelled = [(label, curie) for label in ["positive", "negative"] for curie in abc_curies.get(label, [])]

    def fetch(item):
        return _fetch_reference_features(item[1], mod_abbreviation, profile_name, version,
                                         text_from_reference, cache_dir)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(fetch, labelled))
    else:
        results = [fetch(item) for item in labelled]
    for (label, reference_curie), result in zip(labelled, results):
        if result is None:
            missing += 1
            continue
        pooled, text = result
        if use_bow:
            blocks = []
            if use_embedding_block:
                blocks.append(sp.csr_matrix(pooled.reshape(1, -1)))
            blocks.append(bow_vectorizer.transform([remove_stopwords(text).lower() if text else ""]))
            rows.append(sp.hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0])
        else:
            rows.append(pooled)
        y.append(int(label == "positive"))
        if kept_curies is not None:
            kept_curies.append(reference_curie)
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
                     mod_abbreviation: str = None,
                     embedding_profile: str = ABC_EMBEDDING_PROFILE,
                     strata: Optional[Strata] = None, rebalance: str = "none",
                     embedding_cache_dir: Optional[str] = None, embedding_fetch_workers: int = 1,
                     positive_share: Optional[dict] = None, fixed_model: Optional[tuple] = None):
    """``strata`` (SCRUM-6482) gives each training reference its NNC bucket: the
    holdout test set is then also scored per bucket and weighted to the corpus
    mix, and ``rebalance="bucket_mix"`` trains with those corpus weights.
    ``positive_share`` ({bucket: estimated share of positives in the corpus}) adds a
    holdout score weighted to the estimated corpus (bucket mix and positives within
    each bucket) and a decision threshold tuned for it, and
    ``rebalance="bucket_and_positive_mix"`` trains with those weights
    (``..._balanced``: the same, with both classes scaled to equal total weight)."""
    if use_abc_embeddings:
        # New path (SCRUM-5781): use the ABC's precomputed OpenAI embeddings instead
        # of loading BioWordVec and pooling word vectors over downloaded Markdown.
        # BoW (when enabled) is hashed from the parquet's own paragraph text, so no
        # Markdown download is needed.
        logger.info("Loading training set from ABC precomputed embeddings.")
        # BoW is a fixed convention for ABC-embedding models (classify hardcodes it
        # on too), so force it here regardless of use_bow_features — a model trained
        # without it would mismatch feature dims at classify time.
        # A profile name identifies exactly one registered version today; resolve
        # it rather than making the operator pass a matching pair on the CLI.
        # Refuse an unknown name outright: falling back to fulltext would train on
        # the wrong features and stamp a profile that does not exist.
        profile = get_profile_by_name(embedding_profile)
        if profile is None:
            raise ValueError(f"Unknown embedding profile: {embedding_profile}")
        logger.info(f"Training on the '{profile.name}' embedding profile (v{profile.version}).")
        kept_curies: list = []
        X, y = _build_abc_embedding_features(abc_curies or {}, mod_abbreviation, use_bow=True,
                                             profile_name=profile.name, version=profile.version,
                                             kept_curies=kept_curies, cache_dir=embedding_cache_dir,
                                             workers=embedding_fetch_workers)
        logger.info("Finished loading training set.")
        logger.info(f"Dataset size: {str(len(y))}")
        y = np.array(y)
        buckets = [strata.bucket(curie) for curie in kept_curies] if strata else None
        sample_weight = None
        if rebalance == "bucket_mix":
            if strata is None:
                raise ValueError("rebalance='bucket_mix' needs the NNC strata")
            sample_weight = corpus_weights(buckets, strata)
            weight_of = dict(zip(buckets, sample_weight))
            logger.info("Training with corpus-mix sample weights: "
                        + ", ".join(f"{b} {weight_of[b]:.2f}" for b in scored_buckets(buckets)))
        elif rebalance in POSITIVE_MIX_MODES:
            if strata is None or not positive_share:
                raise ValueError(f"rebalance='{rebalance}' needs the NNC strata and the per-bucket prevalence")
            sample_weight = prevalence_weights(buckets, y, strata, positive_share)
            if rebalance == "bucket_and_positive_mix_balanced":
                sample_weight = balance_classes(sample_weight, y)
            weight_of = dict(zip(zip(buckets, y.tolist()), sample_weight))
            logger.info("Training with estimated-corpus sample weights (+/-): " + ", ".join(
                f"{b} {weight_of.get((b, 1), float('nan')):.2f}/{weight_of.get((b, 0), float('nan')):.2f}"
                for b in scored_buckets(buckets)))
        # The matrix is wide+sparse (embedding + BoW), so the same SVC/MLP skip and
        # gradient-booster concurrency cap the BioWordVec BoW path uses apply here.
        return _select_and_fit_model(X, y, remove_outliers, outlier_method, outlier_contamination,
                                     use_bow_features=True, use_lsh_features=False,
                                     sample_weight=sample_weight, buckets=buckets, strata=strata,
                                     positive_share=positive_share, fixed_model=fixed_model)

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


def _supports_sample_weight(model) -> bool:
    return "sample_weight" in inspect.signature(model.fit).parameters


def _decision_scores(model, X):
    """Positive-class scores: predict_proba where the model has it, else the decision function."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:  # e.g. SGDClassifier with a loss that has no probabilities
            pass
    return model.decision_function(X)


def _best_threshold(y_true, scores, weights):
    """The score cut-off with the best weighted F1 (ties: the higher cut-off)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 501)))
    best_threshold, best_f1 = float(candidates[0]), -1.0
    for threshold in candidates:
        f1 = f1_score(y_true, (scores >= threshold).astype(int), sample_weight=weights, zero_division=0)
        if f1 >= best_f1:
            best_threshold, best_f1 = float(threshold), f1
    return best_threshold, best_f1


def _fit_fold(model, X, y, fit_weight, train_rows, val_rows):
    model = clone(model)
    if fit_weight is not None:
        model.fit(X[train_rows], y[train_rows], sample_weight=fit_weight[train_rows])
    else:
        model.fit(X[train_rows], y[train_rows])
    return val_rows, _decision_scores(model, X[val_rows]), model.predict(X[val_rows])


def _cross_validate(model, X, y, fit_weight, folds, n_jobs):
    """Out-of-fold scores of every row, and precision/recall/F1 (at the model's own cut-off) of every fold,
    with the folds fitted in parallel worker processes (SCRUM-6482)."""
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fit_fold)(model, X, y, fit_weight, train_rows, val_rows)
        for train_rows, val_rows in folds.split(X, y))
    oof = np.zeros(len(y), dtype=float)
    per_fold = []
    for val_rows, scores, preds in results:
        oof[val_rows] = scores
        per_fold.append({"precision": precision_score(y[val_rows], preds, zero_division=0),
                         "recall": recall_score(y[val_rows], preds, zero_division=0),
                         "f1": f1_score(y[val_rows], preds, zero_division=0)})
    return oof, per_fold


def _tuned_threshold_stats(best_classifier, X_train_val, y_train_val, X_test, y_test, idx_train_val, idx_test,
                           fit_weight, buckets, strata, positive_share, folds, oof=None, n_jobs=1):
    """Decision threshold for the estimated corpus (SCRUM-6482), and the holdout scored with it.

    The cut-off is chosen on out-of-fold scores of the train+val rows (the selected model's settings refit per
    fold, with the training weights), weighted to the estimated corpus, so the holdout is not used to choose it.
    Also gives the holdout's precision-recall area (average precision) weighted the same way, which compares
    models independently of any cut-off. ``oof``: out-of-fold scores already computed for these folds."""
    if oof is None:
        oof, _ = _cross_validate(best_classifier, X_train_val, y_train_val, fit_weight, folds, n_jobs)
    train_weights = prevalence_weights([buckets[i] for i in idx_train_val], y_train_val, strata, positive_share)
    threshold, oof_f1 = _best_threshold(y_train_val, oof, train_weights)
    test_weights = prevalence_weights([buckets[i] for i in idx_test], y_test, strata, positive_share)
    test_scores = _decision_scores(best_classifier, X_test)
    tuned_pred = (test_scores >= threshold).astype(int)
    return {"threshold": round(threshold, 4), "out_of_fold_estimated_corpus_f1": round(float(oof_f1), 3),
            "test_estimated_corpus": _weighted_metrics(y_test, tuned_pred, test_weights),
            "test_plain": _weighted_metrics(y_test, tuned_pred, None),
            "test_estimated_corpus_average_precision": round(float(
                average_precision_score(y_test, test_scores, sample_weight=test_weights)), 3)}


def _weighted_metrics(y_true, y_pred, weights):
    return {
        "precision": round(float(precision_score(y_true, y_pred, sample_weight=weights, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, sample_weight=weights, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, sample_weight=weights, zero_division=0)), 3)}


def _bucket_metrics(y_true, y_pred, buckets, strata):
    """Holdout metrics per NNC bucket, and weighted to the corpus mix (SCRUM-6482):
    the labeled pool over-represents NNC-HIGH, so the plain test F1 overstates
    performance on the papers the classifier will actually see."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    by_bucket = {}
    for bucket in scored_buckets(buckets):
        mask = np.array([b == bucket for b in buckets])
        by_bucket[bucket] = {
            "n": int(mask.sum()), "positives": int(y_true[mask].sum()),
            "precision": round(float(precision_score(y_true[mask], y_pred[mask], zero_division=0)), 3),
            "recall": round(float(recall_score(y_true[mask], y_pred[mask], zero_division=0)), 3),
            "f1": round(float(f1_score(y_true[mask], y_pred[mask], zero_division=0)), 3)}
    weights = corpus_weights(buckets, strata)
    weighted = {
        "precision": round(float(precision_score(y_true, y_pred, sample_weight=weights, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, sample_weight=weights, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, sample_weight=weights, zero_division=0)), 3)}
    return by_bucket, weighted


def _select_and_fit_model(X, y, remove_outliers, outlier_method, outlier_contamination,
                          use_bow_features=False, use_lsh_features=False,
                          sample_weight=None, buckets=None, strata=None, positive_share=None,
                          fixed_model=None):
    """``sample_weight`` (per row) trains every candidate with those weights;
    candidates whose ``fit`` takes no weights are skipped. ``buckets`` (per row
    NNC bucket) with ``strata`` adds per-bucket and corpus-weighted holdout
    metrics to the stats (SCRUM-6482). ``fixed_model`` (name, settings) skips the
    model search and trains that one model type with those settings, e.g. to compare
    rebalancing modes on the same model (SCRUM-6482)."""
    if remove_outliers and (sample_weight is not None or buckets is not None):
        raise ValueError("--remove_outliers cannot be combined with NNC rebalancing: it drops rows")

    # Step 1: Split data into train+val (80%) and holdout test set (20%). Splitting
    # the row indices too does not change the split (same rows, same random_state).
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        X, y, np.arange(X.shape[0]), test_size=0.20, stratify=y, random_state=42
    )
    fit_params = {}
    if sample_weight is not None:
        fit_params["sample_weight"] = np.asarray(sample_weight)[idx_train_val]

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

    candidates = POSSIBLE_CLASSIFIERS
    fold_jobs = min(5, int(os.environ.get('SEARCH_MAX_JOBS', '5')))
    oof_scores = None
    if fixed_model is not None:
        candidates = {}
        best_classifier_name, best_params = fixed_model
        best_classifier = clone(POSSIBLE_CLASSIFIERS[best_classifier_name]['model']).set_params(**best_params)
        logger.info(f"No model search: {best_classifier_name} with fixed settings {best_params}; "
                    f"5-fold cross-validation, {fold_jobs} folds in parallel.")
        oof_scores, per_fold = _cross_validate(best_classifier, X_train_val, y_train_val,
                                               fit_params.get("sample_weight"), stratified_k_folds, fold_jobs)
        summaries = (("mean", np.mean), ("std", np.std))
        best_results = {f"{stat}_test_{metric}": [float(fn([fold[metric] for fold in per_fold]))]
                        for metric in ("precision", "recall", "f1") for stat, fn in summaries}
        best_index = 0
        best_penalized_score = best_results["mean_test_f1"][0]
    else:
        logger.info("Starting model selection with hyperparameter optimization and cross-validation.")
        logger.info("Using penalized scoring: weighted_score = 0.7 * test_f1 + 0.3 * cv_f1 - penalty")

    for classifier_name, classifier_info in candidates.items():
        # On the high-dimensional sparse matrix produced by the BoW block, RBF-SVC
        # is slow/weak and MLPClassifier is both memory-heavy (its input-layer
        # weight matrix is n_features x hidden, ~210MB at 2**18 features, multiplied
        # across RandomizedSearchCV's parallel workers -> OOM) and prone to severe
        # overfitting. Skip both when BoW features are enabled.
        if use_bow_features and classifier_name in ('SVC', 'MLPClassifier'):
            logger.info(f"Skipping {classifier_name}: not suitable for the high-dimensional sparse "
                        f"BoW feature matrix.")
            continue
        if fit_params and not _supports_sample_weight(classifier_info['model']):
            logger.info(f"Skipping {classifier_name}: its fit() takes no sample weights.")
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
        random_search.fit(X_train_val, y_train_val, **fit_params)

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
    best_classifier.fit(X_train_val, y_train_val, **fit_params)

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

    stats["sample_weighted"] = sample_weight is not None
    if buckets is not None and strata is not None:
        test_buckets = [buckets[i] for i in idx_test]
        by_bucket, weighted = _bucket_metrics(y_test, final_test_pred, test_buckets, strata)
        stats["test_by_nnc_bucket"] = by_bucket
        stats["test_corpus_weighted"] = weighted
        logger.info(f"Holdout test set weighted to the corpus NNC mix: F1={weighted['f1']:.3f}, "
                    f"Precision={weighted['precision']:.3f}, Recall={weighted['recall']:.3f}")
        for bucket, m in by_bucket.items():
            logger.info(f"  {bucket:7s} n={m['n']:5d} positives={m['positives']:5d} "
                        f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
        if positive_share:
            # also fix the positives within each bucket: the closest the labelled holdout gets to the corpus
            estimated = _weighted_metrics(y_test, final_test_pred,
                                          prevalence_weights(test_buckets, y_test, strata, positive_share))
            stats["test_estimated_corpus_weighted"] = estimated
            logger.info(f"Holdout test set weighted to the estimated corpus (NNC mix and positives per bucket): "
                        f"F1={estimated['f1']:.3f}, Precision={estimated['precision']:.3f}, "
                        f"Recall={estimated['recall']:.3f}")
            logger.info("Tuning the decision threshold for the estimated corpus on out-of-fold scores...")
            tuned = _tuned_threshold_stats(
                best_classifier, X_train_val, y_train_val, X_test, y_test, idx_train_val, idx_test,
                fit_params.get("sample_weight"), buckets, strata, positive_share, stratified_k_folds,
                oof=oof_scores, n_jobs=fold_jobs)
            stats["tuned_threshold"] = tuned
            est = tuned["test_estimated_corpus"]
            logger.info(f"Tuned threshold {tuned['threshold']}: holdout weighted to the estimated corpus "
                        f"F1={est['f1']:.3f}, Precision={est['precision']:.3f}, Recall={est['recall']:.3f}; "
                        f"average precision {tuned['test_estimated_corpus_average_precision']:.3f}")

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
                    training_data_dir: str = None, embedding_recipe: Union[dict, None] = None,
                    data_context: Union[str, None] = None):
    if training_data_dir is None:
        training_data_dir = os.getenv("TRAINING_DIR", "/data/agr_document_classifier/training")
    topic_formatted = topic.replace(':', '_')
    model_filename = f"{mod_abbreviation}_{topic_formatted}_classifier.joblib"
    model_path = os.path.join(training_data_dir, model_filename)

    # Save the classifier directly (compatible with existing classification pipeline)
    joblib.dump(classifier, model_path)
    if test_mode:
        stats_path = os.path.join(training_data_dir, f"{mod_abbreviation}_{topic_formatted}_stats.json")
        with open(stats_path, "w") as stats_file:
            json.dump(stats, stats_file, indent=2, default=str)
        logger.info(f"Saved model to {model_path} and its stats to {stats_path}, "
                    f"skipping upload because in test mode.")
    else:
        upload_ml_model("biocuration_topic_classification", mod_abbreviation=mod_abbreviation, topic=topic,
                        data_novelty=data_novelty, data_context=data_context,
                        production=production,
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
    parser.add_argument("--embedding_profile", type=str, default=ABC_EMBEDDING_PROFILE,
                        choices=registered_profile_names(),
                        help="Which ABC embedding profile to train on. Default is the fulltext "
                             "profile; the abstract profile trains on title+abstract only, for "
                             "classification that must run before a PDF exists.")
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
    # SCRUM-5697. Stamped onto every tag the classifier creates from this model, so
    # the data-context policy lives on the ml_model row rather than in pipeline code.
    # The terms are a hierarchy, not four disjoint options:
    #     ATP:0000323 data context
    #     |-- ATP:0000324 mentioned data
    #     |   |-- ATP:0000360 background information
    #     |   +-- ATP:0000325 experimentally studied data
    #     +-- ATP:0000326 marker data
    #         |-- ATP:0000328 expression marker
    #         +-- ATP:0000327 genetic marker
    # Left as None here and resolved per MOD after parsing: the rule is not
    # universal. Every model this trainer uploads is a topic classifier, and WB's
    # topic tags take the root term (Ceri Van Slyke, SCRUM-5697, 2026-09-01) --
    # but FlyBase asked for ATP:0000325 on all of theirs and the other MODs take
    # the same blanket default, so a fixed ATP:0000323 would have stamped WB's
    # rule onto every MOD. An explicit -C still wins over the per-MOD default.
    parser.add_argument("-C", "--data_context", type=str, required=False, default=None,
                        help="Data context term for the tags this model creates. "
                             "Defaults per MOD: 'ATP:0000323' for WB topic "
                             "classifiers, 'ATP:0000325' otherwise")
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
    parser.add_argument("--require_category", type=str, default=None,
                        help="Train only on references of this category, e.g. Research_Article (the ABC only "
                             "classifies research articles). Default: any category")
    parser.add_argument("--require_mod_reference_type", type=str, default=None,
                        help="Train only on references with this --mod_train reference type, e.g. Experimental "
                             "(WB's classification gate). Default: any type")
    parser.add_argument("--remove_outliers", action="store_true",
                        help="Enable outlier detection and removal")
    parser.add_argument("--outlier_method", type=str, default="isolation_forest",
                        choices=['isolation_forest', 'elliptic_envelope', 'lof'],
                        help="Outlier detection method (default: isolation_forest)")
    parser.add_argument("--outlier_contamination", type=float, default=0.1,
                        help="Expected proportion of outliers (default: 0.1)")
    # SCRUM-6482: rebalance the training set to the corpus mix of the NNC buckets
    parser.add_argument("--nnc_strata_csv", type=str, default=None,
                        help="Per-paper NNC buckets written by scripts/wb_nnc_training_bias.py --papers-csv. "
                             "Adds per-bucket and corpus-weighted holdout metrics; needed by --rebalance.")
    parser.add_argument("--rebalance", type=str, default="none", choices=REBALANCE_MODES,
                        help="'bucket_subsample': drop papers from over-represented NNC buckets until the "
                             "training mix of buckets matches the corpus; 'bucket_mix': keep every paper, weighted "
                             "by corpus share / training share of its bucket (equivalent to repeating papers, "
                             "without leaking repeats across the train/test split or CV folds); "
                             "'bucket_and_positive_mix': as 'bucket_mix', and within each bucket weight positives "
                             "and negatives to the estimated share of positives in the corpus (needs "
                             "--bucket_prevalence_tsv); 'bucket_and_positive_mix_balanced': as "
                             "'bucket_and_positive_mix', with both classes scaled to equal total weight. "
                             "Default: none")
    parser.add_argument("--bucket_prevalence_tsv", type=str, default=None,
                        help="Estimated share of positives per NNC bucket (bucket_prevalence_<model>.tsv from "
                             "scripts/wb_llm_prevalence_estimate.py --nnc_strata_csv). Adds a holdout score "
                             "weighted to the estimated corpus; needed by --rebalance prevalence.")
    parser.add_argument("--rebalance_seed", type=int, default=42,
                        help="Random seed for --rebalance bucket_subsample (default: 42)")
    parser.add_argument("--fixed_model", type=str, default=None, choices=sorted(POSSIBLE_CLASSIFIERS),
                        help="Skip the model search: train only this model type, with --fixed_params "
                             "(e.g. to compare rebalancing modes on one model). Default: search all types")
    parser.add_argument("--fixed_params", type=str, default=None,
                        help="JSON settings for --fixed_model, e.g. '{\"n_estimators\": 177}'")
    parser.add_argument("--embedding_cache_dir", type=str, default=None,
                        help="Keep each reference's fetched embedding and text in this directory (one .npz per "
                             "reference, per profile and version) and read it from there on later runs instead "
                             "of downloading it again. Delete the directory after a re-embedding. Default: no cache")
    parser.add_argument("--embedding_fetch_workers", type=int, default=8,
                        help="References to fetch from the ABC in parallel when loading embeddings (default: 8)")
    args = parser.parse_args()
    if args.rebalance != "none" and not args.nnc_strata_csv:
        parser.error("--rebalance needs --nnc_strata_csv")
    if args.bucket_prevalence_tsv and not args.nnc_strata_csv:
        parser.error("--bucket_prevalence_tsv needs --nnc_strata_csv")
    if args.rebalance in POSITIVE_MIX_MODES and not args.bucket_prevalence_tsv:
        parser.error(f"--rebalance {args.rebalance} needs --bucket_prevalence_tsv")
    if args.remove_outliers and args.nnc_strata_csv:
        parser.error("--remove_outliers cannot be combined with --nnc_strata_csv")
    # SCRUM-5697. Resolve the per-MOD default only when the caller stated
    # nothing, so -C remains an override rather than a suggestion.
    if args.data_context is None:
        args.data_context = default_data_context_for_mod(args.mod_train)
    return args


def _cached_reference_metadata(reference_curie: str, cache_dir: Optional[str]):
    """get_reference_metadata, kept as one JSON file per reference under ``cache_dir/reference_metadata`` when
    ``cache_dir`` is given (SCRUM-6482: the same training set is filtered once per rebalancing mode). A failed
    lookup is not cached. Delete the directory to pick up changed dates, categories or reference types."""
    path = os.path.join(cache_dir, "reference_metadata", reference_curie.replace(":", "_") + ".json") \
        if cache_dir else None
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                meta = json.load(f)
            meta["mod_reference_types"] = [tuple(pair) for pair in meta.get("mod_reference_types") or []]
            return meta
        except (OSError, ValueError):
            pass
    meta = get_reference_metadata(reference_curie)
    if meta is not None and path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.{os.getpid()}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(meta, f)
        os.replace(tmp_path, path)
    return meta


def get_filtered_training_curies(args, training_set):
    """Split a training set into positive/negative reference-curie lists, keeping only the references that pass
    the optional ``--filter_date_before`` cutoff and, when given, ``--require_category`` and
    ``--require_mod_reference_type`` (the papers the ABC actually sends to classification, e.g. WB research
    articles of reference type Experimental). Returns ``{"positive": [...], "negative": [...]}``."""
    reference_ids_positive = [agrkbid for agrkbid, classification_value in training_set["data_training"].items() if
                              classification_value == "positive"]
    reference_ids_negative = [agrkbid for agrkbid, classification_value in training_set["data_training"].items() if
                              classification_value == "negative"]
    require_category = getattr(args, "require_category", None)
    require_type = getattr(args, "require_mod_reference_type", None)
    if not (args.filter_date_before or require_category or require_type):
        return {"positive": reference_ids_positive, "negative": reference_ids_negative}

    filter_date = None
    if args.filter_date_before:
        try:
            filter_date = datetime.strptime(args.filter_date_before, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {args.filter_date_before}. Expected YYYY-MM-DD")
            raise
    logger.info(f"Filtering the training set: published on/after {args.filter_date_before or '-'}, "
                f"category {require_category or 'any'}, {args.mod_train} reference type {require_type or 'any'}")

    all_ids = reference_ids_positive + reference_ids_negative
    cache_dir = getattr(args, "embedding_cache_dir", None)
    with ThreadPoolExecutor(max_workers=8) as pool:
        metadata = dict(zip(all_ids, pool.map(lambda ref_id: _cached_reference_metadata(ref_id, cache_dir),
                                              all_ids)))
    reasons = Counter()

    def _keep(ref_id):
        meta = metadata.get(ref_id) or {}
        # the API returns the enum value (research_article), the DB the name (Research_Article)
        if require_category and (meta.get("category") or "").lower() != require_category.lower():
            reasons["category"] += 1
            return False
        if require_type and (args.mod_train, require_type) not in set(meta.get("mod_reference_types") or []):
            reasons["reference type"] += 1
            return False
        if filter_date:
            # References whose (possibly messy PubMed) date cannot be parsed at all are kept.
            ref_date = parse_reference_date(meta.get("date"))
            if ref_date is not None and ref_date < filter_date:
                reasons["date"] += 1
                return False
        return True

    filtered_positive = [r for r in reference_ids_positive if _keep(r)]
    filtered_negative = [r for r in reference_ids_negative if _keep(r)]
    logger.info(f"Training set filtering complete. Positive: {len(reference_ids_positive)} -> "
                f"{len(filtered_positive)}, Negative: {len(reference_ids_negative)} -> "
                f"{len(filtered_negative)}; dropped by {dict(reasons)}")
    return {"positive": filtered_positive, "negative": filtered_negative}


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
                    data_context=args.data_context,
                    production=args.production,
                    no_data=not args.do_not_flag_no_data, species=args.alternative_species,
                    model_path=os.path.join(training_data_dir,
                                            f"{args.mod_train}_{args.datatype_train.replace(':', '_')}"
                                            f"_classifier.joblib"),
                    stats=stats, dataset_id=training_set["dataset_id"], file_extension="joblib")


def train_and_save_model(args, training_data_dir, training_set, abc_curies=None,
                         strata: Optional[Strata] = None, rebalance_info: Optional[dict] = None):
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
        mod_abbreviation=args.mod_train,
        embedding_profile=args.embedding_profile,
        strata=strata,
        rebalance=getattr(args, "rebalance", "none"),
        embedding_cache_dir=getattr(args, "embedding_cache_dir", None),
        embedding_fetch_workers=getattr(args, "embedding_fetch_workers", 1),
        positive_share=(load_bucket_prevalence(args.bucket_prevalence_tsv, args.datatype_train)
                        if getattr(args, "bucket_prevalence_tsv", None) else None),
        fixed_model=((args.fixed_model, json.loads(args.fixed_params or "{}"))
                     if getattr(args, "fixed_model", None) else None))
    if rebalance_info:
        stats["rebalance"] = rebalance_info
    logger.info(f"Best classifier stats: {str(stats)}")
    # Tag the model (via the ml_model embedding_* columns) so the classifier
    # fetches ABC embeddings for it and rebuilds the identical embedding+BoW
    # vector. Legacy BioWordVec models leave those columns NULL and keep their
    # on-the-fly path.
    # Stamp the profile the model was actually trained on, so classify fetches the
    # matching embeddings. train_classifier has already rejected an unknown name.
    trained_profile = get_profile_by_name(args.embedding_profile)
    embedding_recipe = abc_embedding_recipe(profile_name=trained_profile.name,
                                            version=trained_profile.version)
    save_classifier(classifier=classifier, mod_abbreviation=args.mod_train, topic=args.datatype_train,
                    data_novelty=args.data_novelty,
                    data_context=args.data_context,
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
        return
    strata, rebalance_info = None, None
    if args.nnc_strata_csv:
        abc_curies, strata, rebalance_info = rebalance_training_curies(args, abc_curies)
    train_and_save_model(args, training_data_dir, training_set, abc_curies=abc_curies,
                         strata=strata, rebalance_info=rebalance_info)


def rebalance_training_curies(args, abc_curies):
    """Load the NNC strata for the topic and, for --rebalance bucket_subsample, subsample
    the training curies to the corpus mix (SCRUM-6482). The corpus mix counts the
    papers from the --filter_date_before year on, like the training set. Returns
    (curies, strata, stats entry describing the before/after bucket counts)."""
    year_from = int(args.filter_date_before[:4]) if args.filter_date_before else None
    strata = load_strata(args.nnc_strata_csv, args.datatype_train, year_from=year_from)
    before = bucket_counts(abc_curies, strata)
    logger.info("Corpus NNC mix: " + ", ".join(
        f"{b} {100 * strata.corpus_mix[b]:.1f}%" for b in scored_buckets(strata.corpus_mix)))
    logger.info(f"Training set by NNC bucket: {format_counts(before)}")
    if args.rebalance == "bucket_subsample":
        abc_curies = subsample(abc_curies, strata, seed=args.rebalance_seed)
        logger.info(f"Subsampled to the corpus mix (seed {args.rebalance_seed}): "
                    f"{format_counts(bucket_counts(abc_curies, strata))}")
    info = {"mode": args.rebalance, "nnc_strata_csv": os.path.basename(args.nnc_strata_csv),
            "corpus_mix": {b: round(v, 4) for b, v in strata.corpus_mix.items()},
            "bucket_counts_before": before, "bucket_counts_after": bucket_counts(abc_curies, strata)}
    if args.rebalance == "bucket_subsample":
        info["seed"] = args.rebalance_seed
    if args.bucket_prevalence_tsv:
        shares = load_bucket_prevalence(args.bucket_prevalence_tsv, args.datatype_train)
        logger.info("Estimated share of positives per NNC bucket: " + ", ".join(
            f"{b} {100 * shares[b]:.1f}%" for b in scored_buckets(shares)))
        info["bucket_prevalence_tsv"] = os.path.basename(args.bucket_prevalence_tsv)
        info["estimated_positive_share"] = shares
    return abc_curies, strata, info


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
