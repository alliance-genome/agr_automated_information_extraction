import argparse
import logging
import os
import shutil

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

from utils.abc_utils import get_all_ref_curies, download_tei_files_for_references, get_all_curated_entities, \
    upload_ml_model
from utils.tei_utils import convert_all_tei_files_in_dir_to_txt

logger = logging.getLogger(__name__)


class CustomTokenizer:
    def __init__(self, model_name, additional_tokens=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if additional_tokens:
            self.tokenizer.add_tokens(additional_tokens)
        self.max_length = self.tokenizer.model_max_length

    def __call__(self, text):
        # Split the text into smaller chunks
        tokens = []
        for i in range(0, len(text), self.max_length):
            chunk = text[i:i + self.max_length]
            chunk_tokens = self.tokenizer.tokenize(chunk)
            tokens.extend(chunk_tokens)
        return tokens


def fit_vectorizer_on_agr_corpus(mod_abbreviation: str = None, wipe_download_dir: bool = False):
    download_dir = os.getenv("AGR_CORPUS_DOWNLOAD_DIR", "/tmp/alliance_corpus")
    if wipe_download_dir:
        logger.info(f"Wiping download directory: {download_dir}")
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
        logger.info("No TEI files found in the download directory. Downloading TEI files.")
        logger.info(f"Getting all reference curies for {mod_abbreviation} from the Alliance ABC API.")
        ref_curies = get_all_ref_curies(mod_abbreviation=mod_abbreviation)
        logger.info(f"Downloading TEI files for {len(ref_curies)} references.")
        download_tei_files_for_references(ref_curies, download_dir, mod_abbreviation=mod_abbreviation)
        tei_files_present = True

    if tei_files_present:
        logger.info("TEI files found in the download directory. Converting TEI files to TXT files.")
        convert_all_tei_files_in_dir_to_txt(download_dir)

    logger.info("Downloading list of curated genes and alleles from the Alliance ABC API and adding them to the "
                "tokenizer.")
    curated_genes = get_all_curated_entities(mod_abbreviation=mod_abbreviation, entity_type_str="gene")
    curated_alleles = get_all_curated_entities(mod_abbreviation=mod_abbreviation, entity_type_str="allele")
    logger.info("Loading BioBERT tokenizer and adding curated genes and alleles to it.")
    custom_tokenizer = CustomTokenizer(model_name="dmis-lab/biobert-base-cased-v1.2",
                                       additional_tokens=curated_genes + curated_alleles)
    tfidf_vectorizer = TfidfVectorizer(input='filename', tokenizer=custom_tokenizer)
    text_files = (os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith(".txt"))
    logger.info(f"Fitting TFIDF vectorizer on text files.")
    tfidf_vectorizer.fit(text_files)
    string_vectorizer = TfidfVectorizer(vocabulary=tfidf_vectorizer.vocabulary_, tokenizer=custom_tokenizer)
    dummy_corpus = ["dummy document"]
    string_vectorizer.fit(dummy_corpus)
    logger.info(f"TFIDF vectorizer fitted.")
    return string_vectorizer


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
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("-u", "--upload-to-alliance", action="store_true",
                        help="If set, uploads the vectorizer to the Alliance API")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=None)

    vectorizer = fit_vectorizer_on_agr_corpus(mod_abbreviation=args.mod_abbreviation,
                                              wipe_download_dir=args.wipe_download_dir)
    save_vectorizer_to_file(vectorizer, args.output_path)
    if args.upload_to_alliance:
        stats = {
            "model_name": "TFIDF vectorizer",
            "average_precision": None,
            "average_recall": None,
            "average_f1": None,
            "best_params": None,
        }
        upload_ml_model(task_type="tfidf_vectorization", mod_abbreviation=args.mod_abbreviation, topic=None,
                        model_path=args.output_path, stats=stats, dataset_id=None, file_extension="pkl")
    print(f"TFIDF vectorizer saved to {args.output_path}")


if __name__ == "__main__":
    main()

