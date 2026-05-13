import argparse
import csv
import logging
import os

from utils.abc_utils import (get_curie_from_xref,
                             download_md_files_for_references, download_bib_data_for_references)

logger = logging.getLogger(__name__)

blue_api_base_url = os.environ.get('API_SERVER', "literature-rest.alliancegenome.org")


def download_md_files_from_abc_or_convert_pdf(reference_ids_positive, reference_ids_negative, output_dir,
                                              mod_abbreviation):
    """Populate ``output_dir/{positive,negative}`` with one ``.md`` file per reference.

    Source priority (handled inside :func:`download_md_files_for_references`):

    1. Main MD file from ABC (``converted_merged_main`` / ``.md``).
    2. TEI file from ABC, converted on the fly to Markdown via the shared library.
    3. Server-side on-demand conversion via
       ``GET /reference/referencefile/conversion_request/{curie}`` (PDF or
       nXML in ABC → MD), enabled by ``request_conversion=True``. This
       replaces the older local Grobid PDF→TEI→MD path.
    """
    logger.info(f"Positive MD files download started. Number of files to download: {len(reference_ids_positive)}")
    output_dir_positive = os.path.join(output_dir, "positive")
    download_md_files_for_references(reference_ids_positive, output_dir_positive,
                                     mod_abbreviation, request_conversion=True)
    logger.info(f"Negative MD files download started. Number of files to download: {len(reference_ids_negative)}")
    output_dir_negative = os.path.join(output_dir, "negative")
    download_md_files_for_references(reference_ids_negative, output_dir_negative,
                                     mod_abbreviation, request_conversion=True)

    downloaded_reference_ids_positive = [
        os.path.splitext(name)[0].replace("_", ":") for name in os.listdir(output_dir_positive)
        if name.endswith(".md")
    ]
    downloaded_reference_ids_negative = [
        os.path.splitext(name)[0].replace("_", ":") for name in os.listdir(output_dir_negative)
        if name.endswith(".md")
    ]
    logger.info(f"Downloaded {len(downloaded_reference_ids_positive)} positive MD files.")
    logger.info(f"Downloaded {len(downloaded_reference_ids_negative)} negative MD files.")


def download_prioritized_bib_data(reference_ids_priority_1, reference_ids_priority_2,
                                  reference_ids_priority_3, output_dir, mod_abbreviation):

    logger.info(f"Retrieving biblio info for priority_1 papers: Number of references to retrieve: {len(reference_ids_priority_1)}")
    output_dir_priority_1 = os.path.join(output_dir, "priority_1")
    download_bib_data_for_references(reference_ids_priority_1, output_dir_priority_1, mod_abbreviation)

    logger.info(f"Retrieving biblio info for priority_2 papers: Number of references to retrieve: {len(reference_ids_priority_2)}")
    output_dir_priority_2 = os.path.join(output_dir, "priority_2")
    download_bib_data_for_references(reference_ids_priority_2, output_dir_priority_2, mod_abbreviation)

    logger.info(f"Retrieving biblio info for priority_3 papers: Number of references to retrieve: {len(reference_ids_priority_3)}")
    output_dir_priority_3 = os.path.join(output_dir, "priority_3")
    download_bib_data_for_references(reference_ids_priority_3, output_dir_priority_3, mod_abbreviation)


def download_and_categorize_md_files_from_csv(csv_file, output_dir, mod_abbreviation, start_agrkbid=None):
    os.makedirs(os.path.join(output_dir, "positive"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "negative"), exist_ok=True)

    start_processing = start_agrkbid is None
    reference_ids_positive = []
    reference_ids_negative = []

    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file, delimiter=',')  # Change delimiter to comma
        for row in csv_reader:
            agrkb_id = row.get('AGRKBID')
            label = row.get('Positive/Negative')

            if not agrkb_id:
                xref = row.get('XREF')
                agrkb_id = get_curie_from_xref(xref)
                if not agrkb_id or not label:
                    logger.warning(f"Skipping invalid row: {row}")
                    continue

            if not start_processing:
                if agrkb_id == start_agrkbid:
                    start_processing = True
                else:
                    continue

            category = "positive" if label == "1" else "negative"
            file_name = agrkb_id.replace(":", "_")
            category_dir = os.path.join(output_dir, category)
            md_path = os.path.join(category_dir, f"{file_name}.md")

            if os.path.exists(md_path):
                logger.info(f"Skipping {agrkb_id} as MD file already exists")
                continue

            if category == "positive":
                reference_ids_positive.append(agrkb_id)
            else:
                reference_ids_negative.append(agrkb_id)

    download_md_files_from_abc_or_convert_pdf(reference_ids_positive, reference_ids_negative, output_dir,
                                              mod_abbreviation)


def main():
    parser = argparse.ArgumentParser(description="Download and categorize PDFs from a CSV file")
    parser.add_argument("-m", "--mod-abbreviation", required=True, help="mod abbreviation, eg. FB, WB, SGD, ZFIN, MGI,RGD, XB")
    parser.add_argument("-f", "--csv-file", required=True, help="Path to the input CSV file")
    parser.add_argument("-o", "--output-dir", default="downloaded_files", help="Output directory for downloaded files")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR",
                                                                "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    parser.add_argument("-s", "--start-agrkbid", help="AGRKBID to start processing from")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    download_and_categorize_md_files_from_csv(args.csv_file, out_dir, args.mod_abbreviation, args.start_agrkbid)


if __name__ == '__main__':
    main()
