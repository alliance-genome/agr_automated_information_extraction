import logging

import fasttext
import numpy as np
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_embedding_model(model_path):
    logger.info("Loading embeddings...")
    if model_path.endswith(".vec.bin"):
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    else:
        model = fasttext.load_model(model_path)
    logger.info("Finished loading embeddings.")
    return model


def get_document_embedding(model, document, weighted_average_word_embedding: bool = False,
                           standardize_embeddings: bool = False, normalize_embeddings: bool = False,
                           word_to_index=None):
    # Split the document into words
    words = document.split()
    if isinstance(model, KeyedVectors):
        vocab = set(model.key_to_index.keys())
        valid_words = [word for word in words if word in vocab]
        if not valid_words:
            return np.zeros(model.vector_size)
        embeddings = model[valid_words]
        if word_to_index is None:
            word_to_index = model.key_to_index
    else:
        vocab = set(model.get_words())
        valid_words = [word for word in words if word in vocab]
        if not valid_words:
            return np.zeros(model.get_dimension())
        embeddings = np.array([model.get_word_vector(word) for word in valid_words])
        if word_to_index is None:
            word_to_index = {word: idx for idx, word in enumerate(model.get_words())}

    if embeddings.size == 0:
        return np.zeros(model.get_dimension())

    epsilon = 1e-10
    embeddings_2d = embeddings

    if standardize_embeddings:
        # Standardize the embeddings
        scaler = StandardScaler()
        embeddings_2d = scaler.fit_transform(embeddings_2d)

    if normalize_embeddings:
        # Normalize the embeddings
        norm = np.linalg.norm(embeddings_2d, axis=1, keepdims=True) + epsilon
        embeddings_2d /= norm

    if weighted_average_word_embedding:
        weights = np.array([word_to_index[word] / len(word_to_index) for word in valid_words])
        doc_embedding = np.average(embeddings_2d, axis=0, weights=weights)
    else:
        doc_embedding = np.mean(embeddings_2d, axis=0)
    return doc_embedding
