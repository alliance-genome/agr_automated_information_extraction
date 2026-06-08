import logging

import fasttext
import numpy as np
import scipy.sparse as sp
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# Stateless bag-of-words configuration (SCRUM-6132). A HashingVectorizer built
# from these constants needs no fitting, no vocabulary and no IDF, so the train
# and classify scripts produce identical features without shipping any artifact.
# Keep this in one place: it is the single source of truth both scripts share.
BOW_CONFIG = {
    "n_features": 2 ** 18,
    "ngram_range": (1, 1),
    "alternate_sign": False,
    "norm": "l2",
    "binary": False,
    "lowercase": True,
    "token_pattern": r"(?u)\b[\w-]{2,}\b",
}


# Stateless LSH bag-of-concepts configuration (SCRUM-6132 follow-up). Instead of
# hashing token *strings* (BOW_CONFIG) or fitting k-means concept clusters, we
# hash each in-vocabulary word's *embedding* with random-hyperplane LSH
# (Charikar SimHash): similar vectors collide into the same bucket on purpose,
# so synonyms/near-synonyms land in one bag bin without any clustering. The
# random projection is a pure function of (seed, n_bits, n_tables, dim), so it
# is regenerated identically at train and classify time -- nothing is fitted and
# no artifact is shipped, exactly like the hashing BoW block. ``n_tables``
# independent projections are concatenated to reduce the chance that a single
# random hyperplane splits a dense concept region.
LSH_CONFIG = {
    "n_bits": 12,      # 2**12 = 4096 buckets per table
    "n_tables": 4,     # independent projections, concatenated -> 4 * 4096 = 16384 dims
    "seed": 6132,      # fixed -> reproducible across train/classify, nothing shipped
}

# Cache the random projection per embedding dimension: it is deterministic, so it
# only needs to be materialised once per process rather than per document.
_LSH_PROJECTION_CACHE: dict = {}


def load_embedding_model(model_path):
    logger.info("Loading embeddings...")
    if model_path.endswith(".vec.bin"):
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    else:
        model = fasttext.load_model(model_path)
    logger.info("Finished loading embeddings.")
    return model


def get_bow_vectorizer():
    """Return a fresh, stateless HashingVectorizer built from BOW_CONFIG."""
    return HashingVectorizer(**BOW_CONFIG)


def lsh_feature_width():
    """Width of the LSH bag-of-concepts block (``n_tables * 2**n_bits``)."""
    return LSH_CONFIG["n_tables"] * (1 << LSH_CONFIG["n_bits"])


def get_lsh_projection(dim):
    """Return the (cached) random-hyperplane projection for an embedding of size
    ``dim``: an array of shape ``(n_tables, n_bits, dim)`` drawn from a fixed
    seed. Deterministic, so train and classify produce identical buckets.
    """
    key = (dim, LSH_CONFIG["n_bits"], LSH_CONFIG["n_tables"], LSH_CONFIG["seed"])
    projection = _LSH_PROJECTION_CACHE.get(key)
    if projection is None:
        rng = np.random.default_rng(LSH_CONFIG["seed"])
        projection = rng.standard_normal((LSH_CONFIG["n_tables"], LSH_CONFIG["n_bits"], dim))
        _LSH_PROJECTION_CACHE[key] = projection
    return projection


def _raw_embeddings(model, valid_words):
    """Return the raw (un-preprocessed) embedding matrix for ``valid_words``.

    LSH hashes the model's own word vectors so a given word always falls in the
    same concept bucket, independent of the document it appears in (per-document
    standardisation/normalisation would make a word's bucket context-dependent).
    """
    if isinstance(model, KeyedVectors):
        return model[valid_words]
    return np.array([model.get_word_vector(word) for word in valid_words])


def _lsh_histogram(raw_embeddings, dim):
    """Return a ``(1, lsh_feature_width())`` sparse row: the count of in-vocab
    words falling in each LSH bucket, across all ``n_tables`` projections.

    Each word's sign pattern under one projection (``sign(R . v)``) packs into an
    ``n_bits`` integer; that integer (offset by the table) is the bucket index.
    """
    projection = get_lsh_projection(dim)
    n_tables, n_bits, _ = projection.shape
    bucket_size = 1 << n_bits
    powers = (1 << np.arange(n_bits))
    codes = []
    for table in range(n_tables):
        projected = raw_embeddings @ projection[table].T          # (n_valid, n_bits)
        bits = (projected > 0).astype(np.int64)
        codes.append(bits @ powers + table * bucket_size)         # (n_valid,)
    codes = np.concatenate(codes)
    columns, counts = np.unique(codes, return_counts=True)
    return sp.csr_matrix(
        (counts.astype(np.float64), columns, np.array([0, len(columns)])),
        shape=(1, n_tables * bucket_size))


def _default_word_to_index(model):
    if isinstance(model, KeyedVectors):
        return model.key_to_index
    return {word: idx for idx, word in enumerate(model.get_words())}


def _valid_word_embeddings(model, document, standardize_embeddings: bool = False,
                           normalize_embeddings: bool = False):
    """Return (embeddings_2d, valid_words, dim) for the in-vocabulary words of
    ``document``, with the same preprocessing the classifier has always used.

    ``embeddings_2d`` is ``None`` when the document has no in-vocabulary words.
    """
    words = document.split()
    if isinstance(model, KeyedVectors):
        dim = model.vector_size
        vocab = set(model.key_to_index.keys())
        valid_words = [word for word in words if word in vocab]
        if not valid_words:
            return None, [], dim
        embeddings = model[valid_words]
    else:
        dim = model.get_dimension()
        vocab = set(model.get_words())
        valid_words = [word for word in words if word in vocab]
        if not valid_words:
            return None, [], dim
        embeddings = np.array([model.get_word_vector(word) for word in valid_words])

    if embeddings.size == 0:
        return None, [], dim

    embeddings_2d = embeddings
    if standardize_embeddings:
        scaler = StandardScaler()
        embeddings_2d = scaler.fit_transform(embeddings_2d)
    if normalize_embeddings:
        epsilon = 1e-10
        norm = np.linalg.norm(embeddings_2d, axis=1, keepdims=True) + epsilon
        embeddings_2d = embeddings_2d / norm
    return embeddings_2d, valid_words, dim


def _pool_mean(embeddings_2d, valid_words, weighted_average_word_embedding, word_to_index):
    if weighted_average_word_embedding:
        weights = np.array([word_to_index[word] / len(word_to_index) for word in valid_words])
        return np.average(embeddings_2d, axis=0, weights=weights)
    return np.mean(embeddings_2d, axis=0)


def get_document_embedding(model, document, weighted_average_word_embedding: bool = False,
                           standardize_embeddings: bool = False, normalize_embeddings: bool = False,
                           word_to_index=None):
    embeddings_2d, valid_words, dim = _valid_word_embeddings(
        model, document, standardize_embeddings=standardize_embeddings,
        normalize_embeddings=normalize_embeddings)
    if embeddings_2d is None:
        return np.zeros(dim)
    if word_to_index is None:
        word_to_index = _default_word_to_index(model)
    return _pool_mean(embeddings_2d, valid_words, weighted_average_word_embedding, word_to_index)


def build_document_features(model, document, *, use_max_pooling: bool = False, use_bow: bool = False,
                            use_lsh: bool = False, weighted_average_word_embedding: bool = False,
                            standardize_embeddings: bool = False, normalize_embeddings: bool = False,
                            word_to_index=None, bow_vectorizer=None):
    """Build the classifier feature vector for a single document.

    The optional blocks are concatenated in a fixed order ``[mean | max? | lsh? | bow?]``:

    - ``mean``: the existing mean-pooled embedding (identical to
      :func:`get_document_embedding`). Always present and always first (the
      classify "valid embedding" gate slices it off the front).
    - ``max`` (``use_max_pooling``): element-wise max over the valid word vectors,
      capturing the strongest activation any token produces — the signal mean
      pooling averages away.
    - ``lsh`` (``use_lsh``): a stateless LSH bag-of-concepts block (see
      :data:`LSH_CONFIG`) — random-hyperplane buckets over the word embeddings,
      so near-synonyms share a bin. Captures concept presence; unlike ``bow`` it
      can only bin words that have an embedding (OOV identifiers have no vector).
    - ``bow`` (``use_bow``): a stateless hashing bag-of-words block (see
      :data:`BOW_CONFIG`) capturing the presence of discriminative tokens,
      including identifiers that are out-of-vocabulary for the embedding.

    Returns a dense ``np.ndarray`` when no sparse block is requested, otherwise a
    sparse ``scipy.sparse`` row (the ``lsh``/``bow`` blocks are high-dimensional).
    """
    embeddings_2d, valid_words, dim = _valid_word_embeddings(
        model, document, standardize_embeddings=standardize_embeddings,
        normalize_embeddings=normalize_embeddings)
    if word_to_index is None:
        word_to_index = _default_word_to_index(model)

    if embeddings_2d is None:
        mean_block = np.zeros(dim)
    else:
        mean_block = _pool_mean(embeddings_2d, valid_words, weighted_average_word_embedding, word_to_index)

    blocks = [mean_block]
    if use_max_pooling:
        max_block = np.zeros(dim) if embeddings_2d is None else embeddings_2d.max(axis=0)
        blocks.append(max_block)
    dense = np.concatenate(blocks)

    sparse_blocks = []
    if use_lsh:
        if valid_words:
            sparse_blocks.append(_lsh_histogram(_raw_embeddings(model, valid_words), dim))
        else:
            sparse_blocks.append(sp.csr_matrix((1, lsh_feature_width())))
    if use_bow:
        if bow_vectorizer is None:
            bow_vectorizer = get_bow_vectorizer()
        sparse_blocks.append(bow_vectorizer.transform([document]))

    if not sparse_blocks:
        return dense
    return sp.hstack([sp.csr_matrix(dense.reshape(1, -1)), *sparse_blocks], format="csr")


def build_feature_matrix(model, documents, *, use_max_pooling: bool = False, use_bow: bool = False,
                         use_lsh: bool = False, weighted_average_word_embedding: bool = False,
                         standardize_embeddings: bool = False, normalize_embeddings: bool = False,
                         word_to_index=None):
    """Build the stacked feature matrix ``X`` for a list of documents.

    Returns a dense ``np.ndarray`` when no sparse block is requested, or a sparse
    CSR matrix when ``use_bow``/``use_lsh`` is set. A single BoW vectorizer is
    shared across the batch.
    """
    if word_to_index is None:
        word_to_index = _default_word_to_index(model)
    bow_vectorizer = get_bow_vectorizer() if use_bow else None
    rows = [
        build_document_features(
            model, document, use_max_pooling=use_max_pooling, use_bow=use_bow, use_lsh=use_lsh,
            weighted_average_word_embedding=weighted_average_word_embedding,
            standardize_embeddings=standardize_embeddings, normalize_embeddings=normalize_embeddings,
            word_to_index=word_to_index, bow_vectorizer=bow_vectorizer)
        for document in documents
    ]
    if use_bow or use_lsh:
        return sp.vstack(rows, format="csr")
    return np.vstack(rows)
