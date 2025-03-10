import os

from sklearn.feature_extraction.text import TfidfVectorizer

from abc_utils import get_all_ref_curies, download_tei_files_for_references


def fit_vectorizer_on_agr_corpus(mod_abbreviation: str = None):
    ref_curies = get_all_ref_curies(mod_abbreviation=mod_abbreviation)
    download_dir = os.getenv("AGR_CORPUS_DOWNLOAD_DIR", "/tmp/alliance_corpus")
    download_tei_files_for_references(ref_curies, download_dir, mod_abbreviation=mod_abbreviation)
    downloaded_files = (os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith(".tei"))
    tfidf_vectorizer = TfidfVectorizer(input='filename')
    tfidf_vectorizer.fit(downloaded_files)
    return tfidf_vectorizer


def save_vectorizer_to_file():
    pass


def load_vectorizer_from_file():
    pass

