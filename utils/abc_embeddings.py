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

# The ABC-embedding recipe is stored on the model as dedicated ``ml_model``
# columns (SCRUM-5781), NOT overloaded into description/parameters. A model with
# ``embedding_profile`` set was trained on ABC embeddings and the classifier
# rebuilds the matching feature vector; a model with it NULL/absent (every model
# trained before this change) keeps the on-the-fly BioWordVec path.


def _l2(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` L2-normalized (unchanged when its norm is 0)."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def abc_embedding_recipe(use_bow: bool = True) -> dict:
    """The ABC-embedding recipe fields to store on the model at train time
    (the ``ml_model`` embedding_* columns + ``use_bow_features``), so the
    classifier can rebuild the identical feature vector."""
    return {
        "embedding_profile": ABC_EMBEDDING_PROFILE,
        "embedding_version": ABC_EMBEDDING_VERSION,
        "embedding_model": ABC_EMBEDDING_MODEL,
        "embedding_dim": ABC_EMBEDDING_DIM,
        "embedding_pooling": ABC_EMBEDDING_POOLING,
        "use_bow_features": use_bow,
    }


def is_abc_embedding_model(model_meta_data: Optional[dict]) -> bool:
    """True if the model's metadata marks it as trained on ABC embeddings
    (``embedding_profile`` set); legacy BioWordVec models have it NULL/absent."""
    return bool((model_meta_data or {}).get("embedding_profile"))


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
