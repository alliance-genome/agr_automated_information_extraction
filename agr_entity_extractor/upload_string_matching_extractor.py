import logging

import dill

from agr_entity_extractor.models import AllianceStringMatchingEntityExtractorConfig, \
    AllianceStringMatchingEntityExtractor
from utils.abc_utils import upload_ml_model, download_abc_model

logger = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Upload the entity extractor model to the Alliance ML API")
    parser.add_argument("-m", "--mod-abbreviation", required=True,
                        help="The MOD abbreviation (e.g., FB, WB, SGD, etc.)")
    parser.add_argument("--min-matches", required=True, help="Minimum number of matches required for an "
                                                             "entity to be extracted")
    parser.add_argument("--tfidf-threshold", required=True, help="TF-IDF threshold for entity extraction")
    parser.add_argument("-t", "--topic", required=True, help="The topic of the model")
    args = parser.parse_args()

    tfidf_vectorizer_model_file_path = (f"/data/agr_entity_extraction/tfidf_vectorization_"
                                        f"{args.mod_abbreviation}_notopic.dpkl")
    download_abc_model(mod_abbreviation=args.mod_abbreviation, topic=None,
                       output_path=tfidf_vectorizer_model_file_path, task_type="tfidf_vectorization")

    tfidf_vectorizer = dill.load(open(tfidf_vectorizer_model_file_path, "rb"))

    entity_extraction_model_file_path = (f"/data/agr_entity_extraction/biocuration_entity_extraction_"
                                         f"{args.mod_abbreviation}_{args.topic.replace(':', '_')}.dpkl")

    # Initialize the model
    config = AllianceStringMatchingEntityExtractorConfig()
    model = AllianceStringMatchingEntityExtractor(
        config=config,
        min_matches=args.min_matches,
        tfidf_threshold=args.tfidf_threshold,
        tokenizer=tfidf_vectorizer.tokenizer,
        vectorizer=tfidf_vectorizer,
        entities_to_extract=None
    )

    # Serialize the model
    with open(entity_extraction_model_file_path, "wb") as file:
        dill.dump(model, file)

    stats = {
        "model_name": "Alliance String Matching Entity Extractor",
        "average_precision": None,
        "average_recall": None,
        "average_f1": None,
        "best_params": None,
    }
    upload_ml_model(task_type="biocuration_entity_extraction", mod_abbreviation=args.mod_abbreviation,
                    model_path=entity_extraction_model_file_path, stats=stats, topic=args.topic, file_extension="dpkl")
    logger.info(f"String Matching Entity Extractor uploaded to the Alliance API for {args.mod_abbreviation}.")


if __name__ == "__main__":
    main()
