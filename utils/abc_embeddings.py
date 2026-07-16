"""Consume the ABC's precomputed reference embeddings in the document classifier
(SCRUM-5781).

The ABC generates one embedding parquet per merged Markdown of a reference and
registers it as a ``referencefile`` with ``file_class == "embedding"`` (producer:
``agr_literature_service`` SCRUM-6141/6142). Exactly one profile exists today:

    classifier_fulltext_paragraph_chunk_refs_excluded_md_cleaned  (version 1,
    model text-embedding-3-small, 1536-d)

This module is the single source of truth for that profile, for the classifier's
feature recipe (mean of the paragraph-level chunk embeddings), and for the
per-model marker that lets the classifier tell a new ABC-embedding model apart
from a legacy BioWordVec model without any ABC schema change.

Only the reading side lives here (``pyarrow``); nothing in this repo generates
embeddings, so ``openai``/``tiktoken`` are not needed.
"""

import logging
from io import BytesIO
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# --- The one existing ABC classifier-embedding profile (SCRUM-6142). Keep these
# in one place: both train and classify import them so the feature recipe they
# agree on can never drift. ---
ABC_EMBEDDING_PROFILE = "classifier_fulltext_paragraph_chunk_refs_excluded_md_cleaned"
ABC_EMBEDDING_VERSION = 1
ABC_EMBEDDING_MODEL = "text-embedding-3-small"
ABC_EMBEDDING_DIM = 1536
# How the per-reference feature vector is built from the parquet: the arithmetic
# mean of the paragraph-level chunk embeddings (the document-level row is ignored).
ABC_EMBEDDING_POOLING = "paragraph_mean"

# The merged-Markdown source a classifier embedding must come from: the main PDF's
# converted text (supplements produce their own embedding files, which we ignore).
MAIN_SOURCE_FILE_CLASS = "converted_merged_main"

# Canonical parquet columns this module relies on (shared contract, see
# ``agr_abc_document_parsers.embeddings.parquet_io``).
_EMBEDDING_COLUMN = "embedding"
_IS_DOCUMENT_LEVEL_COLUMN = "is_document_level"

# Marker written into ``ml_model.description`` at train time and parsed at
# classify time. Its presence is the retrocompat switch: a model whose metadata
# carries the marker consumes ABC embeddings; a model without it (every model
# trained before this change) keeps using the on-the-fly BioWordVec path.
_MARKER_SENTINEL = "[abc_embeddings]"


def format_embedding_marker(profile_name: str = ABC_EMBEDDING_PROFILE,
                            version: int = ABC_EMBEDDING_VERSION,
                            model: str = ABC_EMBEDDING_MODEL,
                            dim: int = ABC_EMBEDDING_DIM,
                            pooling: str = ABC_EMBEDDING_POOLING) -> str:
    """Build the ``ml_model.description`` marker recording that a model was trained
    on ABC embeddings and with which recipe."""
    return (f"{_MARKER_SENTINEL} profile={profile_name} version={version} "
            f"model={model} dim={dim} pooling={pooling}")


def parse_embedding_marker(description: Optional[str]) -> Optional[dict]:
    """Parse an ABC-embedding marker out of a model's ``description``.

    Returns a dict with ``profile_name``/``version``/``model``/``dim``/``pooling``
    when the marker is present, or ``None`` otherwise (i.e. a legacy BioWordVec
    model). Parsing is tolerant: unknown/malformed tokens are ignored and only the
    sentinel is required.
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
    }
    for token in tail.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key == "version" or key == "dim":
            try:
                parsed[key] = int(value)
            except ValueError:
                logger.warning("Ignoring non-integer %s in embedding marker: %r", key, value)
        elif key in ("profile_name", "model", "pooling"):
            parsed[key] = value
    return parsed


def paragraph_mean_from_parquet(parquet_bytes: bytes) -> Optional[np.ndarray]:
    """Return the mean of the paragraph-level chunk embeddings in an ABC embedding
    parquet, as a float32 vector, or ``None`` when there are no paragraph rows.

    Paragraph rows are those with ``is_document_level`` false; the single
    document-level row (the whole-document vector the producer also stores) is
    deliberately excluded so the feature is exactly the average of the paragraphs.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(BytesIO(parquet_bytes),
                          columns=[_EMBEDDING_COLUMN, _IS_DOCUMENT_LEVEL_COLUMN])
    is_document_level = table.column(_IS_DOCUMENT_LEVEL_COLUMN).to_pylist()
    embeddings = table.column(_EMBEDDING_COLUMN).to_pylist()

    paragraph_vectors = [
        vector for vector, is_doc in zip(embeddings, is_document_level)
        if not is_doc and vector is not None
    ]
    if not paragraph_vectors:
        return None
    return np.mean(np.asarray(paragraph_vectors, dtype=np.float32), axis=0)
