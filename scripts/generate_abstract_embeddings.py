"""Embed title+abstract for a labelled reference set and write ABC-format
embedding parquets (SCRUM-5764).

One parquet per reference, one chunk row per parquet: an abstract is a single
~350-token chunk, so the L2-normalized chunk-mean pool degenerates to the
L2-normalized abstract vector. The ``content`` column carries the exact string
from ``utils.abc_embeddings.abstract_chunk_text`` — that is what the hashed BoW
block is built from, so it must match the ABC producer byte-for-byte.

Registration is deliberately not done here: ``embedding_file`` rows cannot be
created over HTTP (the router exposes GET only), so the catalog row must come
from the ABC-internal producer. See the plan's external dependencies.

Usage:
    OPENAI_API_KEY=... python3 scripts/generate_abstract_embeddings.py \\
        -i outdir/labelled_abstracts.json -o outdir/parquets
"""
import argparse
import json
import logging
import os
import sys
from typing import Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq

from utils.abc_embeddings import ABSTRACT_PROFILE, abstract_chunk_text

logger = logging.getLogger(__name__)

# Batch size for the embeddings endpoint. Abstracts are ~350 tokens, so this is
# well inside the request limit and keeps the run to a handful of calls.
_EMBED_BATCH = 100


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return one embedding per input text, in order, via the OpenAI embeddings
    API. Split out so tests can patch it without touching the network."""
    from openai import OpenAI

    client = OpenAI()
    vectors: List[List[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start:start + _EMBED_BATCH]
        response = client.embeddings.create(model=ABSTRACT_PROFILE.model_name, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        logger.info("Embedded %d/%d", min(start + _EMBED_BATCH, len(texts)), len(texts))
    return vectors


def _parquet_bytes(reference_curie: str, vector: List[float], content: str) -> bytes:
    table = pa.table({
        "embedding": pa.array([vector], type=pa.list_(pa.float32())),
        "is_document_level": pa.array([False], type=pa.bool_()),
        "content": pa.array([content], type=pa.string()),
        "reference_curie": pa.array([reference_curie], type=pa.string()),
        "chunk_index": pa.array([0], type=pa.int32()),
        "profile_name": pa.array([ABSTRACT_PROFILE.name], type=pa.string()),
        "chunking_strategy": pa.array(["abstract"], type=pa.string()),
        "section_title": pa.array(["__abstract__"], type=pa.string()),
    })
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer)
    return buffer.getvalue().to_pybytes()


def generate(records: Iterable[dict], output_dir: str) -> int:
    """Embed each record's title+abstract and write ``{curie}.parquet`` into
    ``output_dir``. Returns the number of parquets written."""
    os.makedirs(output_dir, exist_ok=True)
    records = list(records)
    if not records:
        logger.warning("No records to embed; nothing written to %s", output_dir)
        return 0
    texts = [abstract_chunk_text(r.get("title", ""), r.get("abstract", "")) for r in records]
    vectors = embed_texts(texts)
    for record, content, vector in zip(records, texts, vectors):
        path = os.path.join(output_dir, f"{record['curie']}.parquet")
        with open(path, "wb") as handle:
            handle.write(_parquet_bytes(record["curie"], vector, content))
    logger.info("Wrote %d parquet(s) to %s", len(records), output_dir)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-json", required=True,
                        help="labelled_abstracts.json from prepare_zfin_probe_training_set.py")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("-l", "--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, stream=sys.stdout,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with open(args.input_json) as handle:
        generate(json.load(handle), args.output_dir)


if __name__ == "__main__":
    main()
