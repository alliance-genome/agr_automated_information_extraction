#!/usr/bin/env python3
"""
Extract TF-IDF vectorizers from WB entity extraction model files.

This script loads the serialized model .dpkl files and extracts/inspects
the TF-IDF vectorizer configurations for each entity type.

Usage:
    python scripts/extract_tfidf_vectorizers.py
    python scripts/extract_tfidf_vectorizers.py --mod WB --topic ATP:0000005
    python scripts/extract_tfidf_vectorizers.py --output-dir /tmp/vectorizers
    python scripts/extract_tfidf_vectorizers.py --save-vectorizers
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import dill

logger = logging.getLogger(__name__)

# Topic ID to entity type mapping
TOPIC_TO_ENTITY_TYPE = {
    "ATP:0000005": "gene",
    "ATP:0000110": "transgenic_allele",
    "ATP:0000006": "allele",
    "ATP:0000027": "strain",
    "ATP:0000123": "species",
}

# Default model directory
DEFAULT_MODEL_DIR = "/data/agr_document_classifier"


def load_model(model_path: str):
    """Load a model from a .dpkl file."""
    logger.info("Loading model from: %s", model_path)
    with open(model_path, "rb") as f:
        model = dill.load(f)
    return model


def extract_vectorizer_info(model, topic: str) -> dict:
    """Extract vectorizer information from a model."""
    info = {
        "topic": topic,
        "entity_type": TOPIC_TO_ENTITY_TYPE.get(topic, "unknown"),
        "tfidf_threshold": getattr(model, "tfidf_threshold", None),
        "min_matches": getattr(model, "min_matches", None),
        "match_uppercase": getattr(model, "match_uppercase", None),
        "has_vectorizer": hasattr(model, "vectorizer") and model.vectorizer is not None,
        "has_tokenizer": hasattr(model, "tokenizer") and model.tokenizer is not None,
        "entities_count": len(getattr(model, "entities_to_extract", []) or []),
        "name_to_curie_count": len(getattr(model, "name_to_curie_mapping", {}) or {}),
    }

    if info["has_vectorizer"]:
        vectorizer = model.vectorizer
        info["vocabulary_size"] = len(getattr(vectorizer, "vocabulary_", {}) or {})
        info["idf_shape"] = (
            vectorizer.idf_.shape if hasattr(vectorizer, "idf_") and vectorizer.idf_ is not None else None
        )
        info["n_features"] = (
            vectorizer.idf_.shape[0] if info["idf_shape"] else None
        )

    return info


def print_vectorizer_info(info: dict):
    """Print vectorizer information in a formatted way."""
    print("\n" + "=" * 60)
    print(f"Topic: {info['topic']} ({info['entity_type']})")
    print("=" * 60)
    print(f"  TF-IDF Threshold:    {info['tfidf_threshold']}")
    print(f"  Min Matches:         {info['min_matches']}")
    print(f"  Match Uppercase:     {info['match_uppercase']}")
    print(f"  Has Vectorizer:      {info['has_vectorizer']}")
    print(f"  Has Tokenizer:       {info['has_tokenizer']}")
    print(f"  Entities Count:      {info['entities_count']}")
    print(f"  Name→CURIE Mappings: {info['name_to_curie_count']}")

    if info["has_vectorizer"]:
        print(f"  Vocabulary Size:     {info['vocabulary_size']}")
        print(f"  IDF Shape:           {info['idf_shape']}")
        print(f"  N Features:          {info['n_features']}")


def print_sample_vocabulary(model, num_samples: int = 10):
    """Print sample vocabulary entries with their IDF values."""
    if not hasattr(model, "vectorizer") or model.vectorizer is None:
        print("  No vectorizer available.")
        return

    vectorizer = model.vectorizer
    vocab = getattr(vectorizer, "vocabulary_", {})
    idf = getattr(vectorizer, "idf_", None)

    if not vocab:
        print("  Vocabulary is empty.")
        return

    print(f"\n  Sample Vocabulary ({min(num_samples, len(vocab))} entries):")
    print("  " + "-" * 50)

    for i, (token, idx) in enumerate(list(vocab.items())[:num_samples]):
        idf_val = idf[idx] if idf is not None and idx < len(idf) else "N/A"
        if isinstance(idf_val, float):
            print(f"    {token:30s} idx={idx:8d}  idf={idf_val:.4f}")
        else:
            print(f"    {token:30s} idx={idx:8d}  idf={idf_val}")


def print_sample_entities(model, num_samples: int = 10):
    """Print sample entities from the model."""
    entities = getattr(model, "entities_to_extract", []) or []
    if not entities:
        print("  No entities to extract.")
        return

    entities_list = list(entities) if isinstance(entities, set) else entities
    print(f"\n  Sample Entities ({min(num_samples, len(entities_list))} of {len(entities_list)}):")
    print("  " + "-" * 50)

    for ent in entities_list[:num_samples]:
        mapping = getattr(model, "name_to_curie_mapping", {}) or {}
        curie = mapping.get(ent, "N/A")
        print(f"    {ent:30s} → {curie}")


def save_vectorizer(model, output_path: str):
    """Save the vectorizer to a separate file."""
    if not hasattr(model, "vectorizer") or model.vectorizer is None:
        logger.warning("No vectorizer to save.")
        return False

    logger.info("Saving vectorizer to: %s", output_path)
    with open(output_path, "wb") as f:
        dill.dump(model.vectorizer, f)
    return True


def find_model_files(model_dir: str, mod: str = "WB") -> list:
    """Find all model files for a given MOD."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        logger.error("Model directory does not exist: %s", model_dir)
        return []

    pattern = f"biocuration_entity_extraction_{mod}_ATP_*.dpkl"
    files = list(model_dir.glob(pattern))
    return sorted(files)


def extract_topic_from_filename(filename: str) -> str:
    """Extract topic ID from model filename."""
    # Format: biocuration_entity_extraction_WB_ATP_0000005.dpkl
    basename = Path(filename).stem
    parts = basename.split("_")
    # Find ATP part and reconstruct topic
    for i, part in enumerate(parts):
        if part == "ATP" and i + 1 < len(parts):
            return f"ATP:{parts[i + 1]}"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Extract TF-IDF vectorizers from WB entity extraction models"
    )
    parser.add_argument(
        "-m", "--mod",
        default="WB",
        help="MOD abbreviation (default: WB)"
    )
    parser.add_argument(
        "-t", "--topic",
        help="Specific topic to extract (e.g., ATP:0000005). If not specified, extracts all."
    )
    parser.add_argument(
        "-d", "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing model files (default: {DEFAULT_MODEL_DIR})"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory to save extracted vectorizers (requires --save-vectorizers)"
    )
    parser.add_argument(
        "-s", "--save-vectorizers",
        action="store_true",
        help="Save vectorizers to separate files"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of sample vocabulary/entity entries to show (default: 10)"
    )
    parser.add_argument(
        "-l", "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout
    )

    # Find model files
    if args.topic:
        # Single topic specified
        topic_safe = args.topic.replace(":", "_")
        model_path = Path(args.model_dir) / f"biocuration_entity_extraction_{args.mod}_{topic_safe}.dpkl"
        if not model_path.exists():
            logger.error("Model file not found: %s", model_path)
            sys.exit(1)
        model_files = [model_path]
    else:
        # Find all model files for the MOD
        model_files = find_model_files(args.model_dir, args.mod)
        if not model_files:
            logger.error("No model files found in %s for MOD=%s", args.model_dir, args.mod)
            sys.exit(1)

    logger.info("Found %d model file(s)", len(model_files))

    # Create output directory if saving
    if args.save_vectorizers:
        output_dir = Path(args.output_dir) if args.output_dir else Path("./extracted_vectorizers")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_dir)

    # Process each model file
    all_info = []
    for model_path in model_files:
        topic = extract_topic_from_filename(str(model_path))
        logger.info("Processing: %s (topic=%s)", model_path.name, topic)

        try:
            model = load_model(str(model_path))
            info = extract_vectorizer_info(model, topic)
            all_info.append(info)

            print_vectorizer_info(info)
            print_sample_vocabulary(model, args.samples)
            print_sample_entities(model, args.samples)

            if args.save_vectorizers:
                entity_type = TOPIC_TO_ENTITY_TYPE.get(topic, "unknown")
                output_path = output_dir / f"vectorizer_{args.mod}_{entity_type}.dpkl"
                if save_vectorizer(model, str(output_path)):
                    print(f"\n  Vectorizer saved to: {output_path}")

        except Exception as e:
            logger.error("Failed to process %s: %s", model_path, e)
            continue

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Topic':<20} {'Entity Type':<20} {'Threshold':<12} {'Min Matches':<12} {'Vocab Size':<12}")
    print("-" * 76)
    for info in all_info:
        print(
            f"{info['topic']:<20} "
            f"{info['entity_type']:<20} "
            f"{str(info['tfidf_threshold']):<12} "
            f"{str(info['min_matches']):<12} "
            f"{str(info.get('vocabulary_size', 'N/A')):<12}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
