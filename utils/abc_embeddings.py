"""Consume the ABC's precomputed reference embeddings in the document classifier
(SCRUM-5781).

The ABC generates one embedding parquet per merged Markdown of a reference and
registers it as a ``referencefile`` with ``file_class == "embedding"`` (producer:
``agr_literature_service`` SCRUM-6141/6142). Registered profiles:

    classifier_fulltext_paragraph_chunk_refs_excluded_md_cleaned  (version 1,
    model text-embedding-3-small, 1536-d) — the production profile; one embedding
    per paragraph chunk of the main PDF's converted Markdown.

    classifier_abstract_title_abstract_single_chunk  (version 1, same model and
    dim) — SCRUM-5764; a single chunk of title+abstract, for classification that
    must run before any PDF exists. Note the dims are identical, so the profile
    name is the only thing standing between a mismatched fetch and silently wrong
    predictions — see :func:`get_profile`.

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
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Where a profile's BoW text comes from. Embedding profiles carry their own text in
# the parquet's ``content`` column; a BoW-only profile has no parquet at all and
# reads title+abstract straight off the reference record.
TEXT_SOURCE_PARQUET = "embedding_parquet"
TEXT_SOURCE_REFERENCE_ABSTRACT = "reference_abstract"


@dataclass(frozen=True)
class EmbeddingProfile:
    """One classifier feature profile: which text the model was trained on, which
    feature blocks it expects, and how to recognise its parquet (if it has one).

    A profile is the single switch that decides how the classifier rebuilds a
    reference's feature vector, so train and classify can never disagree.

    ``use_embedding`` is False for a profile with no dense block at all — the ZFIN
    molecular-probe evaluation (SCRUM-5764) found the embedding block added nothing
    over hashed BoW, so the shipping profile drops it, and with it the OpenAI
    dependency and the need for any embedding parquet. Such a profile has
    ``dim == 0`` and ``model_name is None``.

    ``text_source`` says where the BoW text is read from — see ``TEXT_SOURCE_*``.

    ``required_source_file_class`` is the ``file_class`` the embedding's ``source``
    referencefile must have. It is ``None`` for profiles with no source file at
    all — an abstract embedding is derived from the reference record, so
    ``embedding_file.source_referencefile_id`` is NULL (the column is nullable
    precisely to allow this) and there is nothing to match against.
    """

    name: str
    version: int
    model_name: Optional[str]
    dim: int
    required_source_file_class: Optional[str]
    text_source: str = TEXT_SOURCE_PARQUET
    use_embedding: bool = True


# --- The fulltext profile (SCRUM-6142): one embedding per paragraph chunk of the
# main PDF's converted Markdown. This is what every model in production today
# was trained on. ---
FULLTEXT_PROFILE = EmbeddingProfile(
    name="classifier_fulltext_paragraph_chunk_refs_excluded_md_cleaned",
    version=1,
    model_name="text-embedding-3-small",
    dim=1536,
    required_source_file_class="converted_merged_main",
)

# --- The abstract profile (SCRUM-5764): a single chunk holding title+abstract,
# for classification that must happen before any PDF exists. Same model and dim
# as the fulltext profile, which is exactly why the profile name matters: a
# mismatched fetch would line up dimensionally and predict silent nonsense. ---
ABSTRACT_PROFILE = EmbeddingProfile(
    name="classifier_abstract_title_abstract_single_chunk",
    version=1,
    model_name="text-embedding-3-small",
    dim=1536,
    required_source_file_class=None,
)

# --- The abstract BoW-only profile (SCRUM-5764): hashed bag-of-words over
# title+abstract read from the reference record, with no dense block.
#
# This is the profile the ZFIN molecular-probe classifier ships on. The evaluation
# (docs/superpowers/reports/2026-08-06-scrum-5764-evaluation-results.md) measured
# BoW-only at P 0.984 / R 0.980 out-of-fold against 0.976 / 0.972 for embedding+BoW,
# and all 15 curator-annotated hard negatives correct. Because there is no embedding
# block, a model on this profile needs no OpenAI key, no embedding parquet, and no
# dependency on the abstract-embedding pipeline (SCRUM-6140). ---
ABSTRACT_BOW_PROFILE = EmbeddingProfile(
    name="classifier_abstract_title_abstract_bow_only",
    version=1,
    model_name=None,
    dim=0,
    required_source_file_class=None,
    text_source=TEXT_SOURCE_REFERENCE_ABSTRACT,
    use_embedding=False,
)

_PROFILES = {(p.name, p.version): p
             for p in (FULLTEXT_PROFILE, ABSTRACT_PROFILE, ABSTRACT_BOW_PROFILE)}

# Legacy aliases. Every existing call site and every model with
# embedding_profile NULL resolves to the fulltext profile, unchanged.
ABC_EMBEDDING_PROFILE = FULLTEXT_PROFILE.name
ABC_EMBEDDING_VERSION = FULLTEXT_PROFILE.version
ABC_EMBEDDING_MODEL = FULLTEXT_PROFILE.model_name
ABC_EMBEDDING_DIM = FULLTEXT_PROFILE.dim
# How the dense per-reference vector is built: L2-normalize each chunk embedding,
# average them, and L2-normalize the mean (SCRUM-6052 recipe). The document-level
# parquet row is ignored. Applies to every profile.
ABC_EMBEDDING_POOLING = "l2_chunk_mean"
# The merged-Markdown source a fulltext classifier embedding must come from: the
# main PDF's converted text (supplements produce their own embedding files, which
# we ignore). Kept as a module constant because it is also the strict fallback for
# an unregistered profile name.
MAIN_SOURCE_FILE_CLASS = FULLTEXT_PROFILE.required_source_file_class


def get_profile(profile_name: str, version: int) -> Optional[EmbeddingProfile]:
    """Return the registered profile for ``(profile_name, version)``, or ``None``
    when the pair is unknown. Callers must treat ``None`` as "refuse to guess":
    profiles can share dimensions, so falling back to an arbitrary profile would
    produce silently wrong features rather than an error."""
    return _PROFILES.get((profile_name, version))


def get_profile_by_name(profile_name: str) -> Optional[EmbeddingProfile]:
    """Return the registered profile with ``profile_name``, or ``None``.

    A profile name identifies exactly one registered version today, so an operator
    naming a profile on the CLI does not also have to supply a matching version.
    Resolving through the registry (rather than a literal tuple at the call site)
    keeps a future third profile from being silently invisible to the trainer.
    """
    return next((p for p in _PROFILES.values() if p.name == profile_name), None)


def registered_profile_names() -> list:
    """Every registered profile name, sorted. The trainer's ``--embedding_profile``
    choices come from here so registering a profile is the only edit needed to make
    it selectable."""
    return sorted({p.name for p in _PROFILES.values()})


def abstract_chunk_text(title: str, abstract: str) -> str:
    """Return the exact chunk text embedded (and hashed for BoW) by the abstract
    profile: title and abstract joined by a blank line, each stripped, with empty
    parts omitted so no stray separator survives.

    This is the coordination point with the ABC producer (SCRUM-6140). The hashed
    BoW block is built from this same string, and hashing is exact-token, so the
    format must match production byte-for-byte or a model trained here will get a
    different BoW block at classify time. Keep it defined only in this function.
    """
    parts = [part.strip() for part in (title or "", abstract or "")]
    return "\n\n".join(part for part in parts if part)


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


def abc_embedding_recipe(profile_name: str = ABC_EMBEDDING_PROFILE,
                         version: int = ABC_EMBEDDING_VERSION) -> dict:
    """The (profile, version) to store on the model at train time — the only pair
    the classifier needs to select which stored embedding to fetch for a reference.
    Pooling (L2 chunk-mean) and the BoW block are fixed conventions applied to every
    ABC-embedding model, and model/dim are read from the parquet, so none of those
    are stored on the model."""
    return {
        "embedding_profile": profile_name,
        "embedding_version": version,
    }


def is_abc_embedding_model(model_meta_data: Optional[dict]) -> bool:
    """True if the model's metadata marks it as profile-driven (``embedding_profile``
    set); legacy BioWordVec models have it NULL/absent.

    Note this gates the whole non-BioWordVec feature path, not the presence of an
    embedding specifically: a BoW-only profile (``use_embedding=False``) answers True
    here because its features are still rebuilt from its profile rather than by
    pooling word vectors over downloaded Markdown."""
    return bool((model_meta_data or {}).get("embedding_profile"))


def profile_pair_from_model(model_meta_data: Optional[dict]) -> Tuple[str, int]:
    """Return the ``(embedding_profile, embedding_version)`` an ABC-embedding model
    was trained against, defaulting to the fulltext pair when the model does not
    carry one — which is every model uploaded before the abstract profile existed.
    """
    metadata = model_meta_data or {}
    profile_name = metadata.get("embedding_profile") or ABC_EMBEDDING_PROFILE
    version = metadata.get("embedding_version")
    return profile_name, ABC_EMBEDDING_VERSION if version is None else int(version)


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
