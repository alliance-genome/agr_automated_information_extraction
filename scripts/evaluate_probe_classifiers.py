"""Three-arm evaluation for ZFIN molecular-probe abstract classification
(SCRUM-5764).

Arms, on one fixed stratified split:

* ``embedding+bow`` — the supervised ship candidate. There is deliberately no
  embeddings-only arm: BoW is always concatenated (SCRUM-5781 decision 2, and
  ``_build_abc_embedding_features`` forces it).
* ``bow_only`` — diagnostic. If it matches ``embedding+bow``, a shippable model
  needs no OpenAI key and no abstract embedding profile at all.
* ``llm`` — a small chat model reading the abstract with the curator's
  definition, no training data.

Features are built with the production reader (``paragraph_pool_and_text``) over
the parquets, so what is measured here is what would ship.
"""
import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from utils.abc_embeddings import abstract_chunk_text, paragraph_pool_and_text
from utils.embedding import get_bow_vectorizer
from utils.get_documents import remove_stopwords

logger = logging.getLogger(__name__)


def holdout_indices(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the 80/20 stratified holdout ``_select_and_fit_model`` performs
    internally (``test_size=0.20, stratify=y, random_state=42``) so every arm is
    scored on exactly the same references."""
    return train_test_split(np.arange(len(y)), test_size=0.20, stratify=y, random_state=42)


def build_matrix(records: List[dict], parquet_dir: Optional[str], use_embedding: bool,
                 use_bow: bool) -> Tuple[sp.csr_matrix, List[int], List[dict]]:
    """Build ``(X, y, kept_records)``, mirroring ``_build_abc_embedding_features``:
    L2-normalized chunk-mean pool for the dense block, hashed BoW for the sparse one.

    With ``parquet_dir`` set, both blocks come from the abstract embedding parquets
    and references with no readable parquet are skipped. With ``parquet_dir=None``
    (BoW only) the text is taken straight from the records via
    ``abstract_chunk_text`` — which is byte-identical to the parquet's ``content``
    column, so the BoW baseline is reproducible with no embeddings and no OpenAI
    key at all.

    ``kept_records`` is returned to keep row order aligned with record order — the
    LLM arm scores the same rows.
    """
    if not (use_embedding or use_bow):
        raise ValueError("At least one of use_embedding / use_bow must be set")
    if use_embedding and not parquet_dir:
        raise ValueError("use_embedding requires a parquet_dir; embeddings only "
                         "exist in the parquets")
    bow_vectorizer = get_bow_vectorizer() if use_bow else None
    rows, y, kept = [], [], []
    for record in records:
        if parquet_dir:
            path = os.path.join(parquet_dir, f"{record['curie']}.parquet")
            if not os.path.exists(path):
                logger.warning("No parquet for %s; skipping", record["curie"])
                continue
            with open(path, "rb") as handle:
                result = paragraph_pool_and_text(handle.read())
            if result is None:
                logger.warning("Unreadable parquet for %s; skipping", record["curie"])
                continue
            pooled, text = result
        else:
            pooled, text = None, abstract_chunk_text(record.get("title", ""),
                                                     record.get("abstract", ""))
        blocks = []
        if use_embedding:
            blocks.append(sp.csr_matrix(pooled.reshape(1, -1)))
        if use_bow:
            blocks.append(bow_vectorizer.transform([remove_stopwords(text).lower() if text else ""]))
        rows.append(sp.hstack(blocks, format="csr"))
        y.append(int(record["label"] == "positive"))
        kept.append(record)
    return sp.vstack(rows, format="csr"), y, kept


def _scores(y_true, y_pred, y_prob) -> dict:
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_test": int(len(y_true)),
        "n_positive_test": int(sum(y_true)),
    }
    if y_prob is not None:
        out["average_precision"] = float(average_precision_score(y_true, y_prob))
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        # The operational question: how much recall survives at the precision
        # ZFIN needs before auto-setting "won't curate"?
        for target in (0.90, 0.95, 0.99):
            usable = [(p, r, t) for p, r, t in zip(precision, recall, thresholds) if p >= target]
            best = max(usable, key=lambda item: item[1]) if usable else None
            out[f"recall_at_precision_{int(target * 100)}"] = (
                {"recall": float(best[1]), "threshold": float(best[2])} if best else None)
    return out


def hard_negative_breakdown(test_records: List[dict], y_pred) -> Optional[dict]:
    """Score only the negatives Ceri annotated (``note`` non-empty — in practice
    'string "probe" in abstract').

    These are the references a keyword rule gets wrong: the word "probe" appears
    but the paper still has curatable data. Overall precision hides them because
    they are a small slice, yet they are exactly the population the auto-tagging
    gate must not mislabel, so they get their own number.
    """
    indices = [i for i, record in enumerate(test_records)
               if (record.get("note") or "").strip() and record["label"] == "negative"]
    if not indices:
        return None
    false_positives = sum(1 for i in indices if int(y_pred[i]) == 1)
    return {
        "n_hard_negatives": len(indices),
        "false_positives": int(false_positives),
        "specificity": float((len(indices) - false_positives) / len(indices)),
        "curies": [test_records[i]["curie"] for i in indices],
    }


def out_of_fold_probabilities(records: List[dict], parquet_dir: Optional[str],
                              use_embedding: bool, use_bow: bool,
                              model_name: str = "LGBMClassifier",
                              n_splits: int = 5, seed: int = 42):
    """Return ``(y_true, y_prob, kept)`` with a held-out probability for *every*
    reference, via stratified k-fold.

    The single 80/20 holdout that ``run_supervised`` reports is the right basis for
    comparing arms, but it puts only a couple of the curator-annotated hard
    negatives in the test set — far too few to say anything about the population
    that actually matters. Every reference gets an out-of-fold prediction here, so
    the hard-negative number is computed on all of them.

    A single model class is refit per fold rather than rerunning the full
    ``RandomizedSearchCV``; this is a robustness check on a specific slice, not the
    headline metric, and the search would cost 11 models x n_splits.
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    from agr_document_classifier.models import POSSIBLE_CLASSIFIERS

    X, y_list, kept = build_matrix(records, parquet_dir,
                                   use_embedding=use_embedding, use_bow=use_bow)
    y = np.array(y_list)
    estimator = POSSIBLE_CLASSIFIERS[model_name]["model"]
    y_prob = np.zeros(len(y), dtype=float)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(folds.split(X, y), start=1):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        if hasattr(model, "predict_proba"):
            y_prob[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        else:
            y_prob[test_idx] = 1.0 / (1.0 + np.exp(-model.decision_function(X[test_idx])))
        logger.info("Out-of-fold %d/%d done", fold, n_splits)
    return y, y_prob, kept


def run_supervised(records, parquet_dir, use_embedding, use_bow) -> dict:
    """Fit via the trainer's own model selection and score on its internal holdout.

    ``_select_and_fit_model`` is handed the full matrix on purpose: it performs the
    80/20 stratified split itself. ``holdout_indices`` reproduces that split so we
    can pull probabilities for the precision/recall-threshold analysis and so the
    LLM arm can be scored on the same references.
    """
    from agr_document_classifier.agr_document_classifier_trainer import _select_and_fit_model

    X, y_list, kept = build_matrix(records, parquet_dir,
                                   use_embedding=use_embedding, use_bow=use_bow)
    y = np.array(y_list)
    model, stats = _select_and_fit_model(X, y, False, "isolation_forest", 0.1,
                                         use_bow_features=use_bow, use_lsh_features=False)

    _train_idx, test_idx = holdout_indices(y)
    X_test, y_test = X[test_idx], y[test_idx]
    y_pred = model.predict(X_test)
    # LinearSVC has no predict_proba; the production classify path falls back to
    # decision_function -> sigmoid, so mirror that here for the PR curve.
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = 1.0 / (1.0 + np.exp(-model.decision_function(X_test)))
    else:
        y_prob = None

    result = _scores(y_test, y_pred, y_prob)
    result["selected_model"] = stats.get("model_name")
    result["cv_f1"] = stats.get("average_f1")
    result["n_features"] = int(X.shape[1])
    result["n_records_used"] = len(kept)
    result["hard_negatives"] = hard_negative_breakdown([kept[i] for i in test_idx], y_pred)
    return result


# Ceri Van Slyke's class definition, from SCRUM-5764. The "only" is the whole
# difficulty: a paper that develops a probe AND reports zebrafish biology is a
# negative, because it still has curatable data.
_SYSTEM_PROMPT = """You classify zebrafish literature for ZFIN curators, using only the title and abstract.

Label a reference POSITIVE only if the paper is solely about the synthesis, development, or
characterization of a molecular probe (for example a colorimetric, radiometric, or fluorescent
probe) and reports no other curatable zebrafish biology.

Label it NEGATIVE if it reports any other curatable finding — gene function, expression,
phenotype, disease modelling, development — even when a probe is used as a tool, and even when the
word "probe" appears in the abstract.

Reply with JSON only: {"label": "positive"|"negative", "confidence": <0.0-1.0>}"""


def _chat_json(prompt: str, model: str) -> str:
    """Send one classification request and return the raw response text. Split out
    so tests can patch it without touching the network."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _prompt_for(record: dict, few_shot: List[dict]) -> str:
    parts = []
    for example in few_shot:
        parts.append(f"Title: {example['title']}\nAbstract: {example['abstract']}\n"
                     f"Answer: {{\"label\": \"{example['label']}\", \"confidence\": 1.0}}")
    parts.append(f"Title: {record.get('title', '')}\nAbstract: {record.get('abstract', '')}\nAnswer:")
    return "\n\n".join(parts)


def classify_with_llm(record: dict, few_shot: List[dict], model: str) -> Tuple[int, float]:
    """Return ``(label, confidence)`` for one reference. Unparseable output and
    transport failures are both treated as a low-confidence negative so one bad
    verdict cannot abort a full run or inflate the positive count."""
    try:
        raw = _chat_json(_prompt_for(record, few_shot), model)
    except Exception as exc:  # noqa: BLE001 - any API/transport failure
        logger.warning("LLM call failed for %s: %s", record.get("curie"), exc)
        return 0, 0.0
    try:
        verdict = json.loads(raw)
        label = 1 if str(verdict["label"]).strip().lower() == "positive" else 0
        return label, float(verdict.get("confidence", 0.0))
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Unparseable LLM verdict %r: %s", raw, exc)
        return 0, 0.0


def run_llm(test_records: List[dict], few_shot: List[dict], model: str) -> dict:
    y_true, y_pred, y_prob = [], [], []
    for record in test_records:
        label, confidence = classify_with_llm(record, few_shot, model)
        y_true.append(int(record["label"] == "positive"))
        y_pred.append(label)
        # Signed confidence so the PR curve is meaningful: a confident negative
        # must rank below an unsure positive.
        y_prob.append(confidence if label == 1 else 1.0 - confidence)
    result = _scores(np.array(y_true), np.array(y_pred), np.array(y_prob))
    result["model"] = model
    result["hard_negatives"] = hard_negative_breakdown(test_records, np.array(y_pred))
    return result


def evaluate(records: List[dict], parquet_dir: str, output_dir: str,
             llm_model: str = "gpt-5.4-nano", n_few_shot: int = 8,
             skip_llm: bool = False) -> dict:
    """Run the arms on one shared holdout and write the results."""
    os.makedirs(output_dir, exist_ok=True)

    # Establish the shared split from the same kept-records list the supervised
    # arms use, so the LLM arm scores exactly the same references. Both supervised
    # arms keep the same records (each needs a readable parquet), so one call is
    # enough to derive the split.
    _X, y_list, kept = build_matrix(records, parquet_dir, use_embedding=True, use_bow=True)
    y = np.array(y_list)
    train_idx, test_idx = holdout_indices(y)
    train_records = [kept[i] for i in train_idx]
    test_records = [kept[i] for i in test_idx]

    # Few-shot examples come only from the train side, balanced, so the LLM arm
    # never sees a test abstract.
    positives = [r for r in train_records if r["label"] == "positive"][:n_few_shot // 2]
    negatives = [r for r in train_records if r["label"] == "negative"][:n_few_shot // 2]

    arms = {
        "embedding+bow": run_supervised(records, parquet_dir, True, True),
        "bow_only": run_supervised(records, parquet_dir, False, True),
    }
    if skip_llm:
        logger.info("Skipping the LLM arm (--skip-llm).")
    else:
        arms["llm"] = run_llm(test_records, positives + negatives, llm_model)

    results = {
        "n_records_input": len(records),
        "n_records_used": len(kept),
        "n_train": len(train_records),
        "n_test": len(test_records),
        "n_few_shot": len(positives) + len(negatives),
        "arms": arms,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as handle:
        json.dump(results, handle, indent=2)

    lines = ["| arm | precision | recall | F1 | AP | recall@P95 |",
             "| --- | --: | --: | --: | --: | --: |"]
    for name, scores in results["arms"].items():
        at95 = scores.get("recall_at_precision_95")
        at95_text = f"{at95['recall']:.3f} @ {at95['threshold']:.3f}" if at95 else "unreachable"
        lines.append(f"| {name} | {scores['precision']:.3f} | {scores['recall']:.3f} | "
                     f"{scores['f1']:.3f} | {scores.get('average_precision', float('nan')):.3f} | "
                     f"{at95_text} |")
    with open(os.path.join(output_dir, "results.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    logger.info("Results written to %s", output_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-json", required=True)
    parser.add_argument("-p", "--parquet-dir", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--llm-model", default="gpt-5.4-nano")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Run only the supervised arms (no OPENAI_API_KEY needed)")
    parser.add_argument("-l", "--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, stream=sys.stdout,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with open(args.input_json) as handle:
        evaluate(json.load(handle), args.parquet_dir, args.output_dir,
                 llm_model=args.llm_model, skip_llm=args.skip_llm)


if __name__ == "__main__":
    main()
