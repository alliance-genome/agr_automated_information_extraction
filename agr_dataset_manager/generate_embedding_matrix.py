import argparse
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from utils.embedding import get_document_embedding, load_embedding_model
from utils.tei_utils import AllianceTEI


logger = logging.getLogger(__name__)


def process_tei_files(base_folder, all_mods, embedding_model, output_file):
    embeddings = defaultdict(lambda: defaultdict(list))
    processed_files = 0

    if isinstance(embedding_model, KeyedVectors):
        word_to_index = embedding_model.key_to_index
    else:
        word_to_index = {word: idx for idx, word in enumerate(embedding_model.get_words())}

    for mod in all_mods:
        mod_folder = os.path.join(base_folder, mod)
        if not os.path.isdir(mod_folder):
            logger.warning(f"Warning: MOD folder {mod} does not exist.")
            continue

        # Process each file in the MOD folder
        for filename in os.listdir(mod_folder):
            if filename.endswith(".tei"):
                file_path = os.path.join(mod_folder, filename)
                curie = filename.replace(".tei", "").replace("_", ":")

                # Extract fulltext using AllianceTEI
                tei_parser = AllianceTEI()
                try:
                    tei_parser.load_from_file(file_path)
                    fulltext = tei_parser.get_fulltext()
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    continue

                if not fulltext:
                    logger.warning(f"Skipping file {filename} due to missing content.")
                    continue

                # Generate average embedding
                avg_embedding = get_document_embedding(model=embedding_model, document=fulltext,
                                                       word_to_index=word_to_index)

                embeddings[curie][tuple(avg_embedding)].append(mod)
                processed_files += 1
                if processed_files % 1000 == 0:
                    logger.info(f"Processed {processed_files} files...")
                    partial_aggregation = aggregate_by_avg_embedding(embeddings, all_mods)
                    save_to_csv(partial_aggregation, output_file)

    final_aggregation = aggregate_by_avg_embedding(embeddings, all_mods)
    save_to_csv(final_aggregation, output_file)


def aggregate_by_avg_embedding(embeddings, all_mods):
    data = []
    for curie, avg_embedding_mods in embeddings.items():
        for avg_embedding, mods in avg_embedding_mods.items():
            entry = {"CURIE": curie}
            for mod_column in all_mods:
                entry[mod_column] = 1 if mod_column in mods else 0
            entry["Average_Embedding"] = np.array(avg_embedding)
            data.append(entry)
    return data


def save_to_csv(aggregated_embedding_data, output_file):
    embedding_dim = len(aggregated_embedding_data[0]["Average_Embedding"])
    embedding_columns = [f"Embedding_{i}" for i in range(embedding_dim)]

    df = pd.DataFrame(aggregated_embedding_data)
    df[embedding_columns] = pd.DataFrame(df["Average_Embedding"].tolist(), index=df.index)
    df = df.drop(columns=["Average_Embedding"])

    # Save the DataFrame to a CSV file
    logger.info(f"Saving embeddings matrix to {output_file}...")
    df.to_csv(output_file, index=False)


# Main function
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate a matrix of average word embeddings for TEI files.")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the base folder containing MOD subfolders with TEI files."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output CSV file with the embedding matrix."
    )
    parser.add_argument(
        "--mods",
        type=str,
        nargs="+",
        default=["WB", "SGD", "MGI", "RGD", "FB", "ZFIN"],
        help="List of MODs to process. Default: WB, SGD, MGI, RGD, FB, ZFIN."
    )
    parser.add_argument(
        "-e", "--embedding_model_path",
        type=str,
        help="Path to the word embedding model"
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR",
                                                                "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_file = args.output_file
    mods = args.mods

    embedding_model = load_embedding_model(args.embedding_model_path)

    # Process TEI files and generate embeddings
    print(f"Processing TEI files in folder: {input_folder}")
    process_tei_files(input_folder, mods, embedding_model, args.output_file)


# Entry point
if __name__ == "__main__":
    main()
