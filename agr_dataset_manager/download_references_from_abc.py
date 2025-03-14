import argparse
import os
import logging
from utils.abc_utils import (
    download_tei_files_for_references,
    download_main_pdf,
    get_pmids_from_reference_curies
)
from utils.tei_utils import convert_all_tei_files_in_dir_to_txt

# Configure the logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_reference_curies(input_file):
    """
    Read a file containing reference CURIEs and return a list of CURIEs.
    """
    try:
        with open(input_file, "r") as file:
            curies = [line.strip() for line in file if line.strip()]
        logger.info(f"Loaded {len(curies)} CURIEs from input file: {input_file}")
        return curies
    except Exception as e:
        logger.error(f"Error reading input file '{input_file}': {e}")
        return []


def download_pdfs(output_directory, mod_abbreviation, curie_list):
    """
    Download PDFs for references and save them in the specified directory.
    Download files only for the specified MOD ID and CURIE list if provided.
    """
    os.makedirs(output_directory, exist_ok=True)
    logger.info(f"Downloading PDFs to {output_directory} for MOD: {mod_abbreviation}...")

    try:
        for curie in curie_list:
            download_main_pdf(file_name=curie.replace(":", "_"), output_dir=output_directory,
                              mod_abbreviation=mod_abbreviation, agr_curie=curie)
            logger.info(f"PDF downloaded for CURIE: {curie}")
    except Exception as e:
        logger.error(f"Error downloading PDFs: {e}")


def download_tei_files(output_directory, mod_abbreviation, curie_list):
    """
    Download TEI files for references and save them in the specified directory.
    Download files only for the specified MOD ID and CURIE list if provided.
    """
    os.makedirs(output_directory, exist_ok=True)
    logger.info(f"Downloading TEI files to {output_directory} for MOD {mod_abbreviation}...")

    try:
        download_tei_files_for_references(reference_curies=curie_list, mod_abbreviation=mod_abbreviation,
                                          output_dir=output_directory)
    except Exception as e:
        logger.error(f"Error downloading TEI files: {e}")


def convert_tei_to_txt(tei_directory):
    """
    Convert TEI files in the specified directory to text files
    and remove the TEI files after conversion.
    """
    logger.info(f"Converting TEI files in {tei_directory} to text files...")

    try:
        convert_all_tei_files_in_dir_to_txt(tei_directory)
        logger.info("TEI files converted to text successfully.")
    except Exception as e:
        logger.error(f"Error converting TEI files: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download reference files from the Alliance using abc_utils."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the downloaded files."
    )
    parser.add_argument(
        "--download",
        choices=["pdf", "tei", "text"],
        required=True,
        help="Specify the type of file to download or save. Options: 'pdf', 'tei', or 'text' (converted from TEI)."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to an input file containing a list of reference CURIEs to download."
    )
    parser.add_argument(
        "--mod-abbreviation",
        required=True,
        help="MOD abbreviation (e.g., MGI, ZFIN, SGD, etc.) to specify the model organism database."
    )

    args = parser.parse_args()
    output_directory = args.output_dir

    # Retrieve CURIE list from input file (if provided)
    curie_list = parse_reference_curies(args.input_file) if args.input_file else None
    logger.info("Converting CURIEs to PMIDs...")
    curie_to_pmid_dict = get_pmids_from_reference_curies(curie_list)

    logger.info(f"Starting process to download {args.download.upper()} files...")
    logger.info(f"Output directory: {output_directory}")
    if curie_list:
        logger.info(f"Number of CURIEs specified: {len(curie_list)}")

    if args.download == "pdf":
        download_pdfs(output_directory, args.mod_abbreviation, curie_list)
        for curie, pmid in curie_to_pmid_dict.items():
            curie_file = os.path.join(output_directory, f"{curie.replace(':', '_')}.pdf")
            pmid_file = os.path.join(output_directory, f"{pmid.replace(':', '_')}.pdf")
            if os.path.exists(curie_file):
                os.rename(curie_file, pmid_file)
                logger.info(f"Renamed {curie_file} to {pmid_file}")

    elif args.download == "tei":
        download_tei_files(output_directory, args.mod_abbreviation, curie_list)
        for curie, pmid in curie_to_pmid_dict.items():
            curie_file = os.path.join(output_directory, f"{curie.replace(':', '_')}.tei")
            pmid_file = os.path.join(output_directory, f"{pmid.replace(':', '_')}.tei")
            if os.path.exists(curie_file):
                os.rename(curie_file, pmid_file)
                logger.info(f"Renamed {curie_file} to {pmid_file}")
    elif args.download == "text":
        # First, download TEI files, then convert them to text files
        download_tei_files(output_directory, args.mod_abbreviation, curie_list)
        for curie, pmid in curie_to_pmid_dict.items():
            curie_file = os.path.join(output_directory, f"{curie.replace(':', '_')}.tei")
            pmid_file = os.path.join(output_directory, f"{pmid.replace(':', '_')}.tei")
            if os.path.exists(curie_file):
                os.rename(curie_file, pmid_file)
                logger.info(f"Renamed {curie_file} to {pmid_file}")
        convert_tei_to_txt(output_directory)
    else:
        logger.error("Invalid option for --download. Use 'pdf', 'tei', or 'text'.")


if __name__ == "__main__":
    main()
