import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from utils.embedding import get_document_embedding, load_embedding_model
from utils.tei_utils import AllianceTEI


def process_tei_files(base_folder, all_mods, embedding_model):
    data = []
    embeddings = defaultdict(lambda: defaultdict(list))

    if isinstance(embedding_model, KeyedVectors):
        word_to_index = embedding_model.key_to_index
    else:
        word_to_index = {word: idx for idx, word in enumerate(embedding_model.get_words())}

    for mod in all_mods:
        mod_folder = os.path.join(base_folder, mod)
        if not os.path.isdir(mod_folder):
            print(f"Warning: MOD folder {mod} does not exist.")
            continue

        # Process each file in the MOD folder
        for filename in os.listdir(mod_folder):
            if filename.endswith(".tei"):
                file_path = os.path.join(mod_folder, filename)
                curie = filename.replace(".tei", "").replace("_", ":")

                # Extract fulltext using AllianceTEI
                tei_parser = AllianceTEI()
                tei_parser.load_from_file(file_path)
                fulltext = tei_parser.get_fulltext()

                if not fulltext:
                    print(f"Skipping file {filename} due to missing content.")
                    continue

                # Generate average embedding
                avg_embedding = get_document_embedding(model=embedding_model, document=fulltext,
                                                       word_to_index=word_to_index)

                embeddings[curie][tuple(avg_embedding)].append(mod)

    for curie, avg_embedding_mods in embeddings.items():
        for avg_embedding, mods in avg_embedding_mods.items():
            entry = {"CURIE": curie}
            for mod_column in all_mods:
                entry[mod_column] = 1 if mod_column in mods else 0
            entry["Average_Embedding"] = np.array(avg_embedding)
            data.append(entry)
    return data


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
    args = parser.parse_args()

    input_folder = args.input_folder
    output_file = args.output_file
    mods = args.mods

    embedding_model = load_embedding_model(args.embedding_model_path)

    # Process TEI files and generate embeddings
    print(f"Processing TEI files in folder: {input_folder}")
    embedding_data = process_tei_files(input_folder, mods, embedding_model)

    # Create DataFrame and expand embeddings into separate columns
    if embedding_data:
        embedding_dim = len(embedding_data[0]["Average_Embedding"])
        embedding_columns = [f"Embedding_{i}" for i in range(embedding_dim)]

        df = pd.DataFrame(embedding_data)
        df[embedding_columns] = pd.DataFrame(df["Average_Embedding"].tolist(), index=df.index)
        df = df.drop(columns=["Average_Embedding"])

        # Save the DataFrame to a CSV file
        print(f"Saving embeddings matrix to {output_file}...")
        df.to_csv(output_file, index=False)
        print("Done!")
    else:
        print("No embeddings were generated. Ensure valid TEI files are in the specified folder.")


# Entry point
if __name__ == "__main__":
    main()
