"""Consume the ABC's precomputed reference embeddings in the document classifier
(SCRUM-5781).

The ABC generates one embedding parquet per merged Markdown of a reference and
registers it as a ``referencefile`` with ``file_class == "embedding"`` (producer:
``agr_literature_service`` SCRUM-6141/6142). Exactly one profile exists today:

    classifier_fulltext_paragraph_chunk_refs_excluded_md_cleaned  (version 1,
    model text-embedding-3-small, 1536-d)

Feature recipe (validated in SCRUM-6052, ``local_tests/openai_embed_test``): the
dense block is the **L2-normalized chunk-mean pool** of the main-PDF paragraph
embeddings — each paragraph vector L2-normalized, averaged, and the mean
L2-normalized again — optionally concatenated with the same stateless hashed
bag-of-words block the BioWordVec classifiers use. In that analysis the embedding
alone underperformed BoW, while embedding+BoW matched the BoW baseline, so the
production models pair the two.

This module is the single source of truth for the profile, the pooling recipe,
the BoW text source, and the per-model marker that lets the classifier tell a new
ABC-embedding model apart from a legacy BioWordVec model with no ABC schema change.

Only the reading side lives here (``pyarrow``); nothing in this repo generates
embeddings, so ``openai``/``tiktoken`` are not needed.
"""

import logging
from io import BytesIO
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --- The one existing ABC classifier-embedding profile (SCRUM-6142). Keep these
# in one place: both train and classify import them so the feature recipe they
# agree on can never drift. ---
ABC_EMBEDDING_PROFILE = "classifier_fulltext_paragraph_chunk_refs_excluded_md_cleaned"
ABC_EMBEDDING_VERSION = 1
ABC_EMBEDDING_MODEL = "text-embedding-3-small"
ABC_EMBEDDING_DIM = 1536
# How the dense per-reference vector is built: L2-normalize each paragraph
# embedding, average them, and L2-normalize the mean (SCRUM-6052 recipe). The
# document-level parquet row is ignored.
ABC_EMBEDDING_POOLING = "l2_chunk_mean"

# The merged-Markdown source a classifier embedding must come from: the main PDF's
# converted text (supplements produce their own embedding files, which we ignore).
MAIN_SOURCE_FILE_CLASS = "converted_merged_main"

# Canonical parquet columns this module relies on (shared contract, see
# ``agr_abc_document_parsers.embeddings.parquet_io``).
_EMBEDDING_COLUMN = "embedding"
_IS_DOCUMENT_LEVEL_COLUMN = "is_document_level"
_CONTENT_COLUMN = "content"

# Marker written into ``ml_model.description`` at train time and parsed at
# classify time. Its presence is the retrocompat switch: a model whose metadata
# carries the marker consumes ABC embeddings; a model without it (every model
# trained before this change) keeps using the on-the-fly BioWordVec path. The
# ``bow`` field records whether the hashed BoW block was concatenated, so classify
# reconstructs the exact same feature vector without relying on CLI flags.
_MARKER_SENTINEL = "[abc_embeddings]"


def _l2(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` L2-normalized (unchanged when its norm is 0)."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def format_embedding_marker(profile_name: str = ABC_EMBEDDING_PROFILE,
                            version: int = ABC_EMBEDDING_VERSION,
                            model: str = ABC_EMBEDDING_MODEL,
                            dim: int = ABC_EMBEDDING_DIM,
                            pooling: str = ABC_EMBEDDING_POOLING,
                            use_bow: bool = False) -> str:
    """Build the ``ml_model.description`` marker recording that a model was trained
    on ABC embeddings and with which recipe (including whether BoW was used)."""
    return (f"{_MARKER_SENTINEL} profile={profile_name} version={version} "
            f"model={model} dim={dim} pooling={pooling} "
            f"bow={'true' if use_bow else 'false'}")


def parse_embedding_marker(description: Optional[str]) -> Optional[dict]:
    """Parse an ABC-embedding marker out of a model's ``description``.

    Returns a dict with ``profile_name``/``version``/``model``/``dim``/``pooling``/
    ``bow`` when the marker is present, or ``None`` otherwise (i.e. a legacy
    BioWordVec model). Parsing is tolerant: unknown/malformed tokens are ignored
    and only the sentinel is required.
    """
    if not description or _MARKER_SENTINEL not in description:
        return None
    tail = description.split(_MARKER_SENTINEL, 1)[1]
    parsed = {
        "profile_name": ABC_EMBEDDING_PROFILE,
        "version": ABC_EMBEDDING_VERSION,
        "model": ABC_EMBEDDING_MODEL,
        "dim": ABC_EMBEDDING_DIM,
        "pooling": ABC_EMBEDDING_POOLING,
        "bow": False,
    }
    for token in tail.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in ("version", "dim"):
            try:
                parsed[key] = int(value)
            except ValueError:
                logger.warning("Ignoring non-integer %s in embedding marker: %r", key, value)
        elif key == "bow":
            parsed["bow"] = value.strip().lower() == "true"
        elif key in ("profile_name", "model", "pooling"):
            parsed[key] = value
    return parsed


def paragraph_pool_and_text(parquet_bytes: bytes) -> Optional[Tuple[np.ndarray, str]]:
    """Return ``(pooled_vector, paragraph_text)`` for an ABC embedding parquet, or
    ``None`` when it has no paragraph rows.

    ``pooled_vector`` is the L2-normalized mean of the L2-normalized paragraph
    embeddings (the document-level row is excluded). ``paragraph_text`` is the
    concatenation of the paragraph chunks' ``content`` — the (references-excluded)
    document text used to build the hashed BoW block, so a consumer needs only the
    parquet, no extra Markdown download.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(
        BytesIO(parquet_bytes),
        columns=[_EMBEDDING_COLUMN, _IS_DOCUMENT_LEVEL_COLUMN, _CONTENT_COLUMN])
    is_document_level = table.column(_IS_DOCUMENT_LEVEL_COLUMN).to_pylist()
    embeddings = table.column(_EMBEDDING_COLUMN).to_pylist()
    contents = table.column(_CONTENT_COLUMN).to_pylist()

    chunk_vectors = []
    chunk_texts = []
    for embedding, is_doc, content in zip(embeddings, is_document_level, contents):
        if is_doc:
            continue
        if embedding is not None:
            chunk_vectors.append(_l2(np.asarray(embedding, dtype=np.float32)))
        if content:
            chunk_texts.append(content)
    if not chunk_vectors:
        return None
    pooled = _l2(np.mean(chunk_vectors, axis=0)).astype(np.float32)
    return pooled, " ".join(chunk_texts)
