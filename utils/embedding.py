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
                            weighted_average_word_embedding: bool = False,
                            standardize_embeddings: bool = False, normalize_embeddings: bool = False,
                            word_to_index=None, bow_vectorizer=None):
    """Build the classifier feature vector for a single document.

    The optional blocks are concatenated in a fixed order ``[mean | max? | bow?]``:

    - ``mean``: the existing mean-pooled embedding (identical to
      :func:`get_document_embedding`). Always present.
    - ``max`` (``use_max_pooling``): element-wise max over the valid word vectors,
      capturing the strongest activation any token produces — the signal mean
      pooling averages away.
    - ``bow`` (``use_bow``): a stateless hashing bag-of-words block (see
      :data:`BOW_CONFIG`) capturing the presence of discriminative tokens,
      including identifiers that are out-of-vocabulary for the embedding.

    Returns a dense ``np.ndarray`` when ``use_bow`` is False, otherwise a sparse
    ``scipy.sparse`` row (because the BoW block is high-dimensional and sparse).
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

    if not use_bow:
        return dense

    if bow_vectorizer is None:
        bow_vectorizer = get_bow_vectorizer()
    bow_block = bow_vectorizer.transform([document])
    return sp.hstack([sp.csr_matrix(dense.reshape(1, -1)), bow_block], format="csr")


def build_feature_matrix(model, documents, *, use_max_pooling: bool = False, use_bow: bool = False,
                         weighted_average_word_embedding: bool = False,
                         standardize_embeddings: bool = False, normalize_embeddings: bool = False,
                         word_to_index=None):
    """Build the stacked feature matrix ``X`` for a list of documents.

    Returns a dense ``np.ndarray`` (``use_bow`` False) or a sparse CSR matrix
    (``use_bow`` True). A single BoW vectorizer is shared across the batch.
    """
    if word_to_index is None:
        word_to_index = _default_word_to_index(model)
    bow_vectorizer = get_bow_vectorizer() if use_bow else None
    rows = [
        build_document_features(
            model, document, use_max_pooling=use_max_pooling, use_bow=use_bow,
            weighted_average_word_embedding=weighted_average_word_embedding,
            standardize_embeddings=standardize_embeddings, normalize_embeddings=normalize_embeddings,
            word_to_index=word_to_index, bow_vectorizer=bow_vectorizer)
        for document in documents
    ]
    if use_bow:
        return sp.vstack(rows, format="csr")
    return np.vstack(rows)
