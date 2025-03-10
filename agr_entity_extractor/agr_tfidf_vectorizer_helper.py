import os
import argparse

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

