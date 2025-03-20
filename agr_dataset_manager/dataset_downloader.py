import argparse
import csv
import logging
import os

from lxml import etree

from utils.abc_utils import (get_curie_from_xref, download_main_pdf, convert_pdf_with_grobid,
                             download_tei_files_for_references)

logger = logging.getLogger(__name__)

blue_api_base_url = os.environ.get('API_SERVER', "literature-rest.alliancegenome.org")


def check_conversion_failure(tei_content):
    """ Check if the TEI file content indicates a failed conversion. Args: tei_content (str): The TEI file content as
    a string. Returns: bool: True if the conversion failed, False otherwise.
    """
    try:  # Parse the TEI content
        root = etree.fromstring(tei_content)  # Check for empty elements that indicate failure
        title = root.xpath('//tei:title[@level="a"]', namespaces={'tei': 'http://www.tei-c.org/ns/1.0'})
        # Define conditions for failure
        if not title or not title[0].text:
            return True
    except etree.XMLSyntaxError:  # If parsing fails, it indicates a failure
        return True


def download_tei_files_from_abc_or_convert_pdf(reference_ids_positive, reference_ids_negative, output_dir,
                                               mod_abbreviation):
    logger.info(f"Positive tei files download started. Number of files to download: {len(reference_ids_positive)}")
    output_dir_positive = os.path.join(output_dir, "positive")
    download_tei_files_for_references(reference_ids_positive, output_dir_positive, mod_abbreviation)
    logger.info(f"Negative tei files download started. Number of files to download: {len(reference_ids_negative)}")
    output_dir_negative = os.path.join(output_dir, "negative")
    download_tei_files_for_references(reference_ids_negative, output_dir_negative, mod_abbreviation)

    # Count the TEI files in the positive and negative directories
    downloaded_reference_ids_positive = [name[:-4].replace("_", ":") for name in os.listdir(
        output_dir_positive) if name.endswith('.tei')]
    downloaded_reference_ids_negative = [name[:-4].replace("_", ":") for name in os.listdir(
        output_dir_negative) if name.endswith('.tei')]

    logger.info(f"Downloaded {len(downloaded_reference_ids_positive)} positive TEI files.")
    logger.info(f"Downloaded {len(downloaded_reference_ids_negative)} negative TEI files.")

    # after batch download tei file, first check if tei file exist, if not, then download pdf and convert to tei file
    logger.info("Starting PDF download and TEI conversion for files that were not available in TEI format in the ABC.")
    remaining_reference_ids_positive = set(reference_ids_positive) - set(downloaded_reference_ids_positive)
    remaining_reference_ids_negative = set(reference_ids_negative) - set(downloaded_reference_ids_negative)
    agrkb_positive_negative = {reference_curie: "positive" for reference_curie in remaining_reference_ids_positive}
    agrkb_positive_negative.update({reference_curie: "negative" for reference_curie in
                                    remaining_reference_ids_negative})
    for (agrkb_id, category) in agrkb_positive_negative.items():
        file_name = agrkb_id.replace(":", "_")
        category_dir = os.path.join(output_dir, category)
        tei_path = os.path.join(category_dir, f"{file_name}.tei")

        # Check if TEI file already exists
        if os.path.exists(tei_path):
            continue
        else:
            logger.info(f"Downloading PDF for reference {agrkb_id} as {category}")
            pdf_file_path = os.path.join(category_dir, f"{file_name}.pdf")
            download_main_pdf(agrkb_id, mod_abbreviation, file_name, category_dir)
            if not os.path.exists(pdf_file_path):
                logger.error(f"Cannot find {file_name}.pdf")
                continue

            # Convert PDF to TEI
            pdf_content = open(pdf_file_path, "rb")
            logger.info("Converting PDF to TEI")
            response = convert_pdf_with_grobid(pdf_content.read())

            if response.status_code == 200 and not check_conversion_failure(response.content):
                tei_path = os.path.join(category_dir, f"{file_name}.tei")
                with open(tei_path, 'wb') as tei_file:
                    tei_file.write(response.content)
                logger.info(f"Converted {file_name}.pdf to TEI format")
            else:
                logger.error(f"Failed to convert {file_name}.pdf to TEI. Status code: {response.status_code}")
            os.remove(pdf_file_path)


def download_and_categorize_tei_files_from_csv(csv_file, output_dir, mod_abbreviation, start_agrkbid=None):
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
            tei_path = os.path.join(category_dir, f"{file_name}.tei")

            # Check if TEI file already exists
            if os.path.exists(tei_path):
                logger.info(f"Skipping {agrkb_id} as TEI file already exists")
                continue
            else:
                if category == "positive":
                    reference_ids_positive.append(agrkb_id)
                else:
                    reference_ids_negative.append(agrkb_id)

    download_tei_files_from_abc_or_convert_pdf(reference_ids_positive, reference_ids_negative, output_dir,
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

    download_and_categorize_tei_files_from_csv(args.csv_file, out_dir, args.mod_abbreviation, args.start_agrkbid)


if __name__ == '__main__':
    main()
