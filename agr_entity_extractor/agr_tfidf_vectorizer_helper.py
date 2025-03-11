import argparse
import logging
import os
import shutil

from grobid_client.types import TEI
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.abc_utils import get_all_ref_curies, download_tei_files_for_references
from utils.tei_utils import get_sentences_from_tei_section, convert_tei_to_text, convert_all_tei_files_in_dir_to_txt

logger = logging.getLogger(__name__)


def fit_vectorizer_on_agr_corpus(mod_abbreviation: str = None, wipe_download_dir: bool = False):
    download_dir = os.getenv("AGR_CORPUS_DOWNLOAD_DIR", "/tmp/alliance_corpus")
    if wipe_download_dir:
        if os.path.exists(download_dir):
            shutil.rmtree(download_dir)
            logger.info(f"{download_dir} has been wiped.")
            os.makedirs(download_dir, exist_ok=True)

    tei_files_present = False
    txt_files_present = False
    if os.path.exists(download_dir) and len(os.listdir(download_dir)) > 0:
        tei_files_present = len([file for file in os.listdir(download_dir) if file.endswith(".tei")]) > 0
        txt_files_present = len([file for file in os.listdir(download_dir) if file.endswith(".txt")]) > 0

    if not tei_files_present and not txt_files_present:
        ref_curies = get_all_ref_curies(mod_abbreviation=mod_abbreviation)
        download_tei_files_for_references(ref_curies, download_dir, mod_abbreviation=mod_abbreviation)
        tei_files_present = True

    if tei_files_present:
        convert_all_tei_files_in_dir_to_txt(download_dir)
    tfidf_vectorizer = TfidfVectorizer(input='filename')
    text_files = (os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith(".txt"))
    tfidf_vectorizer.fit(text_files)
    return tfidf_vectorizer


def save_vectorizer_to_file(vectorizer, output_path="tfidf_vectorizer.pkl"):
    """Save the fitted TFIDF vectorizer to a file."""
    import pickle
    with open(output_path, "wb") as file:
        pickle.dump(vectorizer, file)


def load_vectorizer_from_file(input_path="tfidf_vectorizer.pkl"):
    """Load a saved TFIDF vectorizer from a file."""
    import pickle
    with open(input_path, "rb") as file:
        return pickle.load(file)


def main():
    """Main function to fit and save a TFIDF vectorizer for the given mod abbreviation."""
    parser = argparse.ArgumentParser(description="Fit and save a TFIDF vectorizer for a corpus.")
    parser.add_argument(
        "-m", "--mod-abbreviation",
        required=True,
        help="MOD abbreviation, e.g., FB, WB, SGD, etc."
    )
    parser.add_argument(
        "-o", "--output-path",
        default="tfidf_vectorizer.pkl",
        help="Output file path to save the vectorizer (default: tfidf_vectorizer.pkl)."
    )
    parser.add_argument("--wipe-download-dir", action="store_true",
                        help="If set, wipes the download directory before processing")
    args = parser.parse_args()

    vectorizer = fit_vectorizer_on_agr_corpus(mod_abbreviation=args.mod_abbreviation,
                                              wipe_download_dir=args.wipe_download_dir)
    save_vectorizer_to_file(vectorizer, args.output_path)
    print(f"TFIDF vectorizer saved to {args.output_path}")


if __name__ == "__main__":
    main()

