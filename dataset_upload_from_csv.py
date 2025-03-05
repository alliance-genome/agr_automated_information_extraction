import csv
import logging
import sys

from abc_utils import get_curie_from_xref, create_dataset, add_entry_to_dataset

logger = logging.getLogger(__name__)


# Function to upload dataset from CSV
def upload_dataset_from_csv(csv_file: str, title: str, description: str, mod_abbreviation: str, topic: str):
    dataset_id, version = create_dataset(title=title, description=description, mod_abbreviation=mod_abbreviation,
                                         topic=topic, dataset_type="document")
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file, delimiter=',')  # Change delimiter to comma
        for row in csv_reader:
            agrkb_id = row.get('AGRKBID')
            positive = "positive" if int(row.get('Positive/Negative')) == 1 else "negative"
            if not agrkb_id:
                xref = row.get('XREF')
                agrkb_id = get_curie_from_xref(xref)
                if not agrkb_id:
                    logger.warning(f"Skipping invalid row: {row}")
                    continue
            add_entry_to_dataset(mod_abbreviation=mod_abbreviation, topic=topic, dataset_type="document",
                                 version=version, reference_curie=agrkb_id, classification_value=positive)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload dataset to ABC from CSV file")
    parser.add_argument("-f", "--csv-file", help="Path to the CSV file")
    parser.add_argument("-t", "--title", help="Name of the dataset")
    parser.add_argument("-d", "--description", help="Description of the dataset")
    parser.add_argument("-m", "--mod-abbreviation", help="Mod abbreviation")
    parser.add_argument("-T", "--topic", help="Topic of the dataset")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )

    upload_dataset_from_csv(args.csv_file, args.title, args.description, args.mod_abbreviation, args.topic)
