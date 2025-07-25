"""
LogisticRegression with balanced dataset sample: under sampling priority 3 papers
"""
import argparse
import copy
import glob
import json
import logging
import os
import os.path
import shutil
import sys
import csv

from pathlib import Path
from typing import Tuple, List
import traceback

import joblib
import nltk
import numpy as np
import requests.exceptions
from gensim.models import KeyedVectors
from grobid_client import Client
from grobid_client.api.pdf import process_fulltext_document
from grobid_client.models import Article, ProcessForm
from grobid_client.types import TEI, File
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

from agr_dataset_manager.dataset_downloader import download_prioritized_bib_data
from utils.abc_utils import download_tei_files_for_references, send_classification_tag_to_abc, \
    get_cached_mod_abbreviation_from_id, download_bib_data_for_references, \
    download_bib_data_for_need_prioritization_references, set_job_success, get_tet_source_id, \
    set_job_started, get_training_set_from_abc, upload_ml_model, download_abc_model, \
    set_job_failure, load_all_jobs, set_indexing_priority
from utils.embedding import load_embedding_model, get_document_embedding
from utils.tei_utils import get_sentences_from_tei_section
from agr_literature_service.lit_processing.utils.report_utils import send_report

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

root_data_path = "/data/agr_document_classifier/"

logger = logging.getLogger(__name__)


def configure_logging(log_level):
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger.handlers = []
    root_logger.addHandler(stream_handler)


def _build_feature_matrix(
    embedding_model,
    training_data_dir: str,
    weighted_average_word_embedding: bool,
    standardize_embeddings: bool,
    normalize_embeddings: bool,
    sections_to_use: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all docs under training_data_dir/{priority_1,priority_2,priority_3},
    embed them, and return (X, y).
    """
    X, y = [], []
    label_mapping = {"priority_1": 0, "priority_2": 1, "priority_3": 2}

    # build word-to-index once
    if isinstance(embedding_model, KeyedVectors):
        word_to_index = embedding_model.key_to_index
    else:
        word_to_index = {w: i for i, w in enumerate(embedding_model.get_words())}

    for label in ["priority_1", "priority_2", "priority_3"]:
        docs = list(get_documents(os.path.join(training_data_dir, label)))
        logger.info(f"  → loading {len(docs)} docs for {label}")
        for _, (_, fulltext, title, abstract) in enumerate(docs, start=1):
            if not sections_to_use:
                text = fulltext
            else:
                parts = []
                if "title" in sections_to_use:
                    parts.append(title)
                if "abstract" in sections_to_use:
                    parts.append(abstract)
                if "fulltext" in sections_to_use:
                    parts.append(fulltext)
                text = " ".join(parts)

            if not text:
                continue

            text = remove_stopwords(text).lower()
            emb = get_document_embedding(
                embedding_model,
                text,
                weighted_average_word_embedding=weighted_average_word_embedding,
                standardize_embeddings=standardize_embeddings,
                normalize_embeddings=normalize_embeddings,
                word_to_index=word_to_index
            )
            X.append(emb)
            y.append(label_mapping[label])

    X = np.array(X)
    y = np.array(y)
    logger.info(f"Built feature matrix X.shape={X.shape}, y.shape={y.shape}")
    return X, y


def train_classifier(
    embedding_model_path: str,
    training_data_dir: str,
    weighted_average_word_embedding: bool = False,
    standardize_embeddings: bool = False,
    normalize_embeddings: bool = False,
    sections_to_use: List[str] = None,
    test_size: float = 0.2,
    n_iter: int = 10,
    cv_folds: int = 3,
    n_jobs: int = 1      # force single‐threaded
) -> Tuple[Pipeline, dict]:

    # --- load embeddings + docs as before into X, y ---
    emb_model = load_embedding_model(model_path=embedding_model_path)
    X, y = _build_feature_matrix(
        emb_model, training_data_dir,
        weighted_average_word_embedding,
        standardize_embeddings,
        normalize_embeddings,
        sections_to_use
    )
    del emb_model

    # --- split once, so we undersample only training set ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42,
        shuffle=True
    )
    logger.info(f"Original train distribution: {np.bincount(y_train)}")

    # ── NEW: undersample P3 **only** in the training fold ──────────────────
    p1_count = int((y_train == 0).sum())
    rus = RandomUnderSampler(
        sampling_strategy={2: p1_count},   # only reduce P3 to match P1
        random_state=42
    )
    X_train, y_train = rus.fit_resample(X_train, y_train)
    logger.info(f"After undersampling P3 in train: {np.bincount(y_train)}")
    # ───────────────────────────────────────────────────────────────────────

    # --- define your pipelines & hyper‐param grids, all with n_jobs=1 ---
    pipelines = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=1))
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', RandomForestClassifier(class_weight='balanced', n_jobs=1))
        ]),
    }
    param_grids = {
        'LogisticRegression': {
            'clf__C': [0.01, 0.1, 1.0],
            'clf__penalty': ['l2'],
            'clf__solver': ['saga']
        },
        'RandomForest': {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [5, 10, None],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2]
        }
    }

    # --- search over these smaller folds/datasets ---
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    best_score = -1
    best_pipe = None
    best_name = None
    best_params = None

    for name, pipe in pipelines.items():
        logger.info(f"→ hyperparameter search for {name}")
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grids[name],
            n_iter=n_iter,
            scoring='f1_macro',
            cv=skf,
            verbose=1,
            n_jobs=1,               # single‐thread CV
            refit=True,
            random_state=42
        )
        search.fit(X_train, y_train)
        logger.info(f"{name} best CV f1_macro = {search.best_score_:.4f}")

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_pipe = search.best_estimator_
            best_name = name
            best_params = search.best_params_

    # --- final evaluation on the hold‐out set ---
    y_pred = best_pipe.predict(X_test)
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1, 2], zero_division=0
    )
    label_rev = {0: "priority_1", 1: "priority_2", 2: "priority_3"}
    per_class = {
        label_rev[i]: {
            "precision": round(float(prec[i]), 3),
            "recall": round(float(rec[i]), 3),
            "f1": round(float(f1[i]), 3),
            "support": int(supp[i])
        }
        for i in range(3)
    }

    stats = {
        "model_name": best_name,
        "average_f1": round(best_score, 3),
        "average_recall": None,
        "average_precision": None,
        "best_params": best_params,
        "per_class_scores": per_class,
        "num_train": len(y_train),
        "num_test": len(y_test),
        "labels": list(label_rev.values())
    }
    return best_pipe, stats


def save_classifier(classifier, mod_abbreviation: str, topic: str, stats: dict, dataset_id: int):
    model_path = f"{root_data_path}training/{mod_abbreviation}_{topic.replace(':', '_')}_classifier.joblib"
    joblib.dump(classifier, model_path)
    upload_ml_model("biocuration_topic_classification", mod_abbreviation=mod_abbreviation, topic=topic,
                    model_path=model_path, stats=stats, dataset_id=dataset_id, file_extension="joblib")


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)


def get_documents(input_docs_dir: str) -> List[Tuple[str, str, str, str]]:
    documents = []
    client = None
    for file_path in glob.glob(os.path.join(input_docs_dir, "*")):
        num_errors = 0
        file_obj = Path(file_path)
        # Process TEI or PDF files as before
        if file_path.endswith(".tei") or file_path.endswith(".pdf"):
            with file_obj.open("rb") as fin:
                if file_path.endswith(".pdf"):
                    if client is None:
                        client = Client(base_url=os.environ.get("GROBID_API_URL"), timeout=1000, verify_ssl=False)
                    logger.info("Started pdf to TEI conversion")
                    form = ProcessForm(
                        segment_sentences="1",
                        input_=File(file_name=file_obj.name, payload=fin, mime_type="application/pdf")
                    )
                    r = process_fulltext_document.sync_detailed(client=client, multipart_data=form)
                    file_stream = r.content
                else:
                    file_stream = fin
                try:
                    article: Article = TEI.parse(file_stream, figures=True)
                except Exception:
                    num_errors += 1
                    continue
                sentences = []
                for section in article.sections:
                    sec_sentences, sec_num_errors = get_sentences_from_tei_section(section)
                    sentences.extend(sec_sentences)
                    num_errors += sec_num_errors
                abstract = ""
                for section in article.sections:
                    if section.name == "ABSTRACT":
                        abs_sentences, num_errors = get_sentences_from_tei_section(section)
                        abstract = " ".join(abs_sentences)
                        break
                documents.append((file_path, " ".join(sentences), article.title, abstract))
        # process plain text files with metadata in key|value format
        elif file_path.endswith(".txt"):
            try:
                with file_obj.open("r", encoding="utf-8") as fin:
                    content = fin.read()
                data = {}
                for line in content.splitlines():
                    if '|' in line:
                        key, value = line.split('|', 1)
                        data[key.strip()] = value.strip()
                title = data.get("title", "")
                abstract = data.get("abstract", "")
                # For fulltext, we combine title and abstract. Should we set it to ""?
                fulltext = f"{title} {abstract}".strip()
                documents.append((file_path, fulltext, title, abstract))
            except Exception as e:
                num_errors += 1
                logger.error(f"Error parsing txt file {file_path}: {e}")
                continue
        if num_errors > 0:
            logger.debug(f"Encountered {num_errors} error(s) while processing {file_path}")
    return documents


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

    for _, (file_path, fulltext, _, _) in enumerate(documents, start=1):
        doc_embedding = get_document_embedding(embedding_model, fulltext, word_to_index=word_to_index)
        X.append(doc_embedding)
        files_loaded.append(file_path)
        valid_embeddings.append(not np.all(doc_embedding == np.zeros_like(doc_embedding)))

    del embedding_model
    X = np.array(X)
    classifications = classifier_model.predict(X)

    # For multi-class, use the probability corresponding to the predicted class
    try:
        predicted_probas = classifier_model.predict_proba(X)
        # zip() function takes the two lists, classifications and predicted_probas,
        # and pairs up their corresponding elements to iterate over both lists simultaneously
        confidence_scores = [probas[pred] for pred, probas in zip(classifications, predicted_probas)]
    except AttributeError:
        # fallback if classifier doesn't support predict_proba
        confidence_scores = [1 / (1 + np.exp(-decision_value))
                             for decision_value in classifier_model.decision_function(X)]
    return files_loaded, classifications, confidence_scores, valid_embeddings


def classify_from_csv_file(csv_path: str, mod_abbr: str, topic: str, embedding_model_path: str):
    output_dir = f"{root_data_path}csv_to_classify"
    os.makedirs(output_dir, exist_ok=True)

    # Read reference IDs from the CSV file
    reference_curies = []
    ref_curie_to_mod_curie = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reference_curies.append(row['id'])
            ref_curie_to_mod_curie[row['id']] = row.get('name', '')

    logger.info(f"Read {len(reference_curies)} reference IDs from {csv_path}")

    # Download bibliographic data and write to txt
    download_bib_data_for_references(reference_curies, output_dir=output_dir, mod_abbreviation=mod_abbr)

    set_priority_for_papers(output_dir, topic, mod_abbr, embedding_model_path, ref_curie_to_mod_curie)


def classify_need_prioritization_papers(mod_abbr: str, topic: str, embedding_model_path: str):
    output_dir = f"{root_data_path}new_to_classify"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    # Fetch all “need prioritization” papers into output_dir
    download_bib_data_for_need_prioritization_references(output_dir, mod_abbr)

    # Only proceed if at least one file was downloaded
    if os.listdir(output_dir):
        set_priority_for_papers(output_dir, topic, mod_abbr, embedding_model_path)
    else:
        logger.info(f"No papers to classify for MOD '{mod_abbr}'.")


def set_priority_for_papers(output_dir: str, topic: str, mod_abbr: str, embedding_model_path: str, ref_curie_to_mod_curie: dict = None):

    # Load classifier and embeddings
    classifier_model_path = f"{root_data_path}training/{mod_abbr}_{topic.replace(':', '_')}_classifier.joblib"
    download_abc_model(mod_abbreviation=mod_abbr, topic=topic, output_path=classifier_model_path,
                       task_type="biocuration_pretriage_priority_classification")
    logger.info(f"Classification model downloaded for mod: {mod_abbr}, topic: {topic}.")

    classifier_model = joblib.load(classifier_model_path)
    embedding_model = load_embedding_model(model_path=embedding_model_path)

    files_loaded, classifications, conf_scores, valid_embeddings = classify_documents(
        input_docs_dir=output_dir,
        embedding_model=embedding_model,
        classifier_model=classifier_model
    )

    label_mapping = {0: "priority_1", 1: "priority_2", 2: "priority_3"}
    results_file = os.path.join(output_dir, "classification_results.csv")
    with open(results_file, mode='w', newline='', encoding='utf-8') as outcsv:
        writer = csv.writer(outcsv)
        headers = ['ReferenceID', 'Classification', 'ConfidenceScore', 'ValidEmbedding']
        if ref_curie_to_mod_curie:
            headers.insert(1, 'MOD_ReferenceID')
        writer.writerow(headers)

        for path, label, score, valid in zip(files_loaded, classifications, conf_scores, valid_embeddings):
            reference_id = Path(path).stem.replace('_', ':')
            row = [reference_id]
            if ref_curie_to_mod_curie:
                row.append(ref_curie_to_mod_curie.get(reference_id, ''))
            row.extend([label_mapping.get(label, 'unknown'), round(score, 4), valid])
            writer.writerow(row)
            priority_name = label_mapping.get(label, 'unknown')
            ref_curie = reference_id
            set_indexing_priority(ref_curie, mod_abbr, priority_name)

    logger.info(f"Classification complete. Results saved to {results_file}")


def save_stats_file(stats, file_path, task_type, mod_abbreviation, topic, version_num, file_extension,
                    dataset_id):
    model_data = {
        "task_type": task_type,
        "mod_abbreviation": mod_abbreviation,
        "topic": topic,
        "version_num": version_num,
        "file_extension": file_extension,
        "model_type": stats.get("model_name"),
        "precision": stats.get("average_precision") or stats.get("average_f1_macro"),
        "recall": stats.get("average_recall"),  # Optional
        "f1_score": stats.get("average_f1") or stats.get("average_f1_macro"),
        "parameters": stats.get("best_params"),
        "dataset_id": dataset_id
    }

    if "per_class_scores" in stats:
        model_data["per_class_scores"] = stats["per_class_scores"]

    with open(file_path, "w") as stats_file:
        json.dump(model_data, stats_file, indent=4)


def parse_arguments():
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
    parser.add_argument("-N", "--skip_training", action="store_true",
                        help="Just upload a pre-existing model and stats file to the ABC without training",
                        required=False)
    parser.add_argument("-l", "--log_level", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="Set the logging level")
    parser.add_argument("--csv_file", type=str, required=False, help="Path to CSV file for flat classification input")

    return parser.parse_args()


def process_classification_jobs(mod_id, topic, jobs, embedding_model):
    mod_abbr = get_cached_mod_abbreviation_from_id(mod_id)
    tet_source_id = get_tet_source_id(mod_abbreviation=mod_abbr, source_method="abc_document_classifier",
                                      source_description="Alliance document classification pipeline using machine "
                                                         "learning to identify papers of interest for curation data "
                                                         "types")
    classifier_file_path = (f"{root_data_path}biocuration_pretriage_priority_classification_{mod_abbr}_"
                            f"{topic.replace(':', '_')}_classifier.joblib")
    try:
        download_abc_model(mod_abbreviation=mod_abbr, topic=topic, output_path=classifier_file_path,
                           task_type="biocuration_pretriage_priority_classification")
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


def process_job_batch(job_batch, mod_abbr, topic, tet_source_id, embedding_model, classifier_model):
    reference_curie_job_map = {job["reference_curie"]: job for job in job_batch}
    prepare_classification_directory()
    download_tei_files_for_references(list(reference_curie_job_map.keys()),
                                      f"{root_data_path}to_classify", mod_abbr)
    files_loaded, classifications, conf_scores, valid_embeddings = classify_documents(
        embedding_model=embedding_model,
        classifier_model=classifier_model,
        input_docs_dir=f"{root_data_path}to_classify")
    send_classification_results(files_loaded, classifications, conf_scores, valid_embeddings, reference_curie_job_map,
                                mod_abbr, topic, tet_source_id)


def prepare_classification_directory():
    os.makedirs(f"{root_data_path}to_classify", exist_ok=True)
    logger.info("Cleaning up existing files in the to_classify directory")
    for file in os.listdir(f"{root_data_path}to_classify"):
        os.remove(os.path.join(f"{root_data_path}to_classify", file))


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
                                                confidence_score=conf_score,
                                                confidence_level=confidence_level, tet_source_id=tet_source_id)
        if result:
            set_job_started(reference_curie_job_map[reference_curie])
            set_job_success(reference_curie_job_map[reference_curie])
        os.remove(file_path)
    logger.info(f"Finished processing batch of {len(files_loaded)} jobs.")


def get_confidence_level(classification, conf_score):
    """
    This function now produces a label that indicates both the priority level and
    the confidence (e.g., "priority_2-High").
    """
    mapping = {0: "priority_1", 1: "priority_2", 2: "priority_3"}
    base_label = mapping.get(classification, "unknown")
    if conf_score < 0.5:
        return f"{base_label}-Low"
    elif conf_score < 0.75:
        return f"{base_label}-Med"
    else:
        return f"{base_label}-High"


def download_training_set(args, training_data_dir):
    training_set = get_training_set_from_abc(mod_abbreviation=args.mod_train, topic=args.datatype_train)
    reference_ids_priority_1 = [
        agrkbid for agrkbid, classification_value in training_set["data_training"].items()
        if classification_value == "priority 1"
    ]
    reference_ids_priority_2 = [
        agrkbid for agrkbid, classification_value in training_set["data_training"].items()
        if classification_value == "priority 2"
    ]
    reference_ids_priority_3 = [
        agrkbid for agrkbid, classification_value in training_set["data_training"].items()
        if classification_value == "priority 3"
    ]

    # Clean up any existing directories
    shutil.rmtree(os.path.join(training_data_dir, "priority_1"), ignore_errors=True)
    shutil.rmtree(os.path.join(training_data_dir, "priority_2"), ignore_errors=True)
    shutil.rmtree(os.path.join(training_data_dir, "priority_3"), ignore_errors=True)

    # Create directories for each priority level
    os.makedirs(os.path.join(training_data_dir, "priority_1"), exist_ok=True)
    os.makedirs(os.path.join(training_data_dir, "priority_2"), exist_ok=True)
    os.makedirs(os.path.join(training_data_dir, "priority_3"), exist_ok=True)

    # Download biblio data for each priority set
    download_prioritized_bib_data(
        reference_ids_priority_1,
        reference_ids_priority_2,
        reference_ids_priority_3,
        output_dir=training_data_dir,
        mod_abbreviation=args.mod_train
    )
    return training_set


def upload_pre_existing_model(args, training_set):
    logger.info("Skipping training. Uploading pre-existing model and stats file to ABC")
    stats = json.load(open(f"{root_data_path}training/{args.mod_train}_"
                           + f"{args.datatype_train.replace(':', '_')}_metadata.json"))
    stats["best_params"] = stats["parameters"]
    stats["model_name"] = stats["model_type"]
    stats["average_precision"] = stats["precision"]
    stats["average_recall"] = stats["recall"]
    stats["average_f1"] = stats["f1_score"]
    upload_ml_model(task_type="biocuration_pretriage_priority_classification", mod_abbreviation=args.mod_train,
                    topic=args.datatype_train,
                    model_path=f"{root_data_path}training/{args.mod_train}_"
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
    training_data_dir = f"{root_data_path}training"
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


def classify_mode(args):
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

    # logger = logging.getLogger(__name__)

    if args.mode == "classify":
        if hasattr(args, 'csv_file') and args.csv_file:
            classify_from_csv_file(
                csv_path=args.csv_file,
                mod_abbr=args.mod_train,
                topic=args.datatype_train,
                embedding_model_path=args.embedding_model_path
            )
        else:
            classify_need_prioritization_papers(
                mod_abbr=args.mod_train,
                topic=args.datatype_train,
                embedding_model_path=args.embedding_model_path)
    else:
        train_mode(args)


if __name__ == '__main__':
    main()
