"""Rebalance a training set to the corpus mix of a stratum (SCRUM-6482).

WB curators validate papers the in-house NNC classifier flagged HIGH far more often than the rest, so a WB
training set over-represents NNC-HIGH papers and under-represents NNC-NEG ones (see
scripts/wb_nnc_training_bias.py). This module brings the training set's per-bucket mix in line with the corpus,
in one of these ways (``--rebalance``):

  - ``bucket_subsample``: drop papers from the over-represented buckets until each bucket has its corpus share.
    The scarcest bucket relative to its share sets the size, so this costs data.
  - ``bucket_mix``: keep every paper and weight it by corpus share / training share of its bucket. Inside each
    bucket the positive/negative split stays as curators labelled it.
  - ``bucket_and_positive_mix``: as ``bucket_mix``, and inside each bucket weight positives and negatives to the
    share of positives estimated for that bucket of the corpus (the LLM reading of a random sample,
    scripts/wb_llm_prevalence_estimate.py --nnc_strata_csv), not the share curators happened to label (e.g. the
    NNC-NEG papers curators validated are far more often positive than NNC-NEG papers are). The training set
    then has the corpus's (low) share of positives.
  - ``bucket_and_positive_mix_balanced``: as ``bucket_and_positive_mix``, then all positives scaled up together
    so both classes carry the same total weight: the positives still look like the corpus's positives and the
    negatives like its negatives, but the model learns from a 50/50 class mix. Its decision threshold then has to
    be set for the real share of positives (the trainer's tuned threshold).

The same per-paper corpus weights also give an estimate of how a model would score on the corpus mix rather
than on the biased labeled pool (``corpus_weights`` on a test split).

The strata come from the per-paper CSV that ``scripts/wb_nnc_training_bias.py --papers-csv`` writes
(reference_curie, topic, nnc_bucket, classification, year): every paper NNC scored for the topic, in the
training set or not. Training papers NNC never scored keep weight 1 and are never dropped.
"""
import csv
import logging
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

REBALANCE_MODES = ("none", "bucket_subsample", "bucket_mix", "bucket_and_positive_mix",
                   "bucket_and_positive_mix_balanced")
POSITIVE_MIX_MODES = ("bucket_and_positive_mix", "bucket_and_positive_mix_balanced")
NO_BUCKET = "no NNC"
LABELS = ("positive", "negative")


@dataclass(frozen=True)
class Strata:
    """The bucket of every scored paper and each bucket's share of the corpus."""
    bucket_of: Dict[str, str]
    corpus_mix: Dict[str, float]

    def bucket(self, curie: str) -> str:
        return self.bucket_of.get(curie, NO_BUCKET)


def load_strata(csv_path: str, topic: str, year_from: Optional[int] = None) -> Strata:
    """Strata for one topic from the papers CSV. The corpus mix counts the papers published in or after
    year_from (papers with no year are kept, as the trainer keeps unparseable dates)."""
    bucket_of: Dict[str, str] = {}
    corpus: Counter = Counter()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["topic"] != topic or row["nnc_bucket"] in ("", NO_BUCKET):
                continue
            bucket_of[row["reference_curie"]] = row["nnc_bucket"]
            year = int(row["year"]) if row.get("year") else None
            if year_from is None or year is None or year >= year_from:
                corpus[row["nnc_bucket"]] += 1
    total = sum(corpus.values())
    if not total:
        raise ValueError(f"{csv_path} has no NNC-scored papers for {topic}")
    return Strata(bucket_of=bucket_of, corpus_mix={b: n / total for b, n in corpus.items()})


def bucket_counts(curies_by_label: Dict[str, List[str]], strata: Strata) -> Dict[str, Dict[str, int]]:
    """{bucket: {"positive": n, "negative": n}} for a training set."""
    counts: Dict[str, Dict[str, int]] = {}
    for label in LABELS:
        for curie in curies_by_label.get(label, []):
            cell = counts.setdefault(strata.bucket(curie), {"positive": 0, "negative": 0})
            cell[label] += 1
    return counts


def subsample(curies_by_label: Dict[str, List[str]], strata: Strata,
              seed: int = 42) -> Dict[str, List[str]]:
    """The largest subset of the training set whose bucket mix matches the corpus mix, drawn at random
    (reproducibly, per seed) within each bucket, so each bucket keeps its positive rate on average.
    Unscored papers are all kept."""
    by_bucket: Dict[str, List[tuple]] = {}
    for label in LABELS:
        for curie in curies_by_label.get(label, []):
            by_bucket.setdefault(strata.bucket(curie), []).append((curie, label))
    scored = {b: papers for b, papers in by_bucket.items() if b != NO_BUCKET and strata.corpus_mix.get(b)}
    if not scored:
        return {label: list(curies_by_label.get(label, [])) for label in LABELS}
    # a bucket missing from the training set cannot be represented; match the mix of the buckets present
    share_total = sum(strata.corpus_mix[b] for b in scored)
    shares = {b: strata.corpus_mix[b] / share_total for b in scored}
    size = min(len(papers) / shares[b] for b, papers in scored.items())
    rng = random.Random(seed)
    kept = list(by_bucket.get(NO_BUCKET, []))
    for bucket in sorted(scored):
        papers = sorted(scored[bucket])
        kept.extend(rng.sample(papers, min(len(papers), int(size * shares[bucket]))))
    result: Dict[str, List[str]] = {label: [] for label in LABELS}
    for curie, label in kept:
        result[label].append(curie)
    return {label: sorted(curies) for label, curies in result.items()}


def corpus_weights(buckets: Sequence[str], strata: Strata) -> np.ndarray:
    """Per-row weights (corpus share / share among these rows of each row's bucket), normalised to mean 1.
    Rows in no bucket, or in a bucket the corpus lacks, get weight 1 before normalisation."""
    counts = Counter(buckets)
    scored_total = sum(n for b, n in counts.items() if b != NO_BUCKET and strata.corpus_mix.get(b))
    present_share = sum(strata.corpus_mix[b] for b in counts if b != NO_BUCKET and strata.corpus_mix.get(b))
    per_bucket: Dict[str, float] = {}
    for bucket, n in counts.items():
        share = strata.corpus_mix.get(bucket) if bucket != NO_BUCKET else None
        per_bucket[bucket] = (share / present_share) / (n / scored_total) if share and scored_total else 1.0
    weights = np.array([per_bucket[b] for b in buckets], dtype=float)
    return weights / weights.mean() if len(weights) else weights


def load_bucket_prevalence(tsv_path: str, topic: str) -> Dict[str, float]:
    """{bucket: estimated share of positives} for one topic, from the positive_share column of the TSV that
    scripts/wb_llm_prevalence_estimate.py --nnc_strata_csv writes."""
    shares: Dict[str, float] = {}
    with open(tsv_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["topic"] == topic:
                shares[row["bucket"]] = float(row["positive_share"])
    if not shares:
        raise ValueError(f"{tsv_path} has no per-bucket prevalence for {topic}")
    return shares


def prevalence_weights(buckets: Sequence[str], labels: Sequence[int], strata: Strata,
                       positive_share: Dict[str, float]) -> np.ndarray:
    """Per-row weights that give every (bucket, class) cell its estimated corpus share: the bucket's corpus share
    times the bucket's estimated share of positives (or of negatives), over the cell's share of these rows.
    Normalised to mean 1. A bucket with no estimate keeps the rows' own positive share in it (a bucket-only
    weight, as ``corpus_weights``); rows in no bucket, or in a bucket the corpus lacks, get weight 1 before
    normalisation. A bucket whose rows are all one class cannot be fixed within the bucket and is weighted as a
    whole."""
    labels = [int(v) for v in labels]
    cells = Counter(zip(buckets, labels))
    rows_in = Counter(buckets)
    in_corpus = [b for b in rows_in if b != NO_BUCKET and strata.corpus_mix.get(b)]
    scored_total = sum(rows_in[b] for b in in_corpus)
    present_share = sum(strata.corpus_mix[b] for b in in_corpus)
    per_cell: Dict[tuple, float] = {}
    for (bucket, label), n in cells.items():
        if bucket not in in_corpus:
            per_cell[(bucket, label)] = 1.0
            continue
        bucket_target = strata.corpus_mix[bucket] / present_share
        own_positive = cells.get((bucket, 1), 0) / rows_in[bucket]
        share = positive_share.get(bucket, own_positive)
        if own_positive in (0.0, 1.0):
            share = own_positive
        target = bucket_target * (share if label == 1 else 1 - share)
        per_cell[(bucket, label)] = target / (n / scored_total)
    weights = np.array([per_cell[cell] for cell in zip(buckets, labels)], dtype=float)
    return weights / weights.mean() if len(weights) else weights


def balance_classes(weights: Sequence[float], labels: Sequence[int]) -> np.ndarray:
    """Scale each class's weights so both classes carry the same total weight (keeping the weights' relative
    sizes within each class), normalised to mean 1."""
    weights = np.asarray(weights, dtype=float)
    labels = np.asarray(labels).astype(int)
    out = weights.copy()
    for label in (0, 1):
        mask = labels == label
        if mask.any() and weights[mask].sum() > 0:
            out[mask] = weights[mask] / weights[mask].sum()
    return out / out.mean() if len(out) else out


def format_counts(counts: Dict[str, Dict[str, int]]) -> str:
    order = ["HIGH", "MEDIUM", "LOW", "NEG", NO_BUCKET]
    parts = []
    for bucket in sorted(counts, key=lambda b: order.index(b) if b in order else len(order)):
        cell = counts[bucket]
        parts.append(f"{bucket} {cell['positive'] + cell['negative']} "
                     f"({cell['positive']}+/{cell['negative']}-)")
    return ", ".join(parts)


def scored_buckets(buckets: Iterable[str]) -> List[str]:
    order = ["HIGH", "MEDIUM", "LOW", "NEG", NO_BUCKET]
    present = set(buckets)
    return [b for b in order if b in present] + sorted(present - set(order))
