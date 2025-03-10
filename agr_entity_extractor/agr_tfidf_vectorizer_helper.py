import argparse
import os

from grobid_client.types import TEI
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.abc_utils import get_all_ref_curies, download_tei_files_for_references
from utils.tei_utils import get_sentences_from_tei_section, convert_tei_to_text


def fit_vectorizer_on_agr_corpus(mod_abbreviation: str = None):
    ref_curies = get_all_ref_curies(mod_abbreviation=mod_abbreviation)
    download_dir = os.getenv("AGR_CORPUS_DOWNLOAD_DIR", "/tmp/alliance_corpus")
    download_tei_files_for_references(ref_curies, download_dir, mod_abbreviation=mod_abbreviation)
    downloaded_files = (os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith(".tei"))

    for tei_file in downloaded_files:
        try:
            with open(tei_file, "rb") as file_stream:
                article = TEI.parse(file_stream, figures=True)
                article_text = convert_tei_to_text(article)
                with open(tei_file.replace(".tei", ".txt"), "w") as text_file:
                    text_file.write(article_text)
                os.remove(tei_file)
        except Exception as e:
            print(f"Error parsing TEI file {tei_file}: {e}")
            continue

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
    args = parser.parse_args()

    vectorizer = fit_vectorizer_on_agr_corpus(mod_abbreviation=args.mod_abbreviation)
    save_vectorizer_to_file(vectorizer, args.output_path)
    print(f"TFIDF vectorizer saved to {args.output_path}")


if __name__ == "__main__":
    main()

