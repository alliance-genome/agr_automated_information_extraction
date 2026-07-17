"""Re-classify previously-classified FB references with the retrained ABC-embedding
models, WITHOUT touching workflow/job tags (SCRUM-5781 final step).

Unlike the normal classification pipeline (which is driven by "classification
needed" workflow jobs), this reads the references that were *already* classified
and just records the new models' output:

- The four TET topics (disease ATP:0000152, physical interaction ATP:0000069,
  new allele ATP:0000006, new transgenic construct ATP:0000013) -> create NEW
  ``topic_entity_tag`` rows (new ``ml_model_id``); the old BioWordVec TETs are
  left in place for provenance. No ``set_job_*`` / workflow-tag changes.
- "No genetic data" (ATP:0000207) -> its result lives in ``manual_indexing_tag``,
  one row per (mod, reference, tag). We UPDATE the existing rows'
  ``confidence_score`` (the prediction column) in place and leave
  ``validation_by_biocurator`` (the independent curator column) untouched, so no
  curator work is lost. No ``ATP:0000359`` workflow tag is created.

Reference sets come from the existing classification records:

- The 4 TET topics share one set — the references that completed FB *reference
  classification*, i.e. those carrying any of the per-data-type "classification
  complete" workflow tags. On FB the three data-type completes at 7427 refs
  (ATP:0000167 / ATP:0000244 / ATP:0000307) union to 7427 references, verified to
  be a strict superset of the new-allele completes (ATP:0000290 / ATP:0000298),
  the overall "complete" tag (ATP:0000169, 7366) and the 61 "in progress" refs —
  so this union is exactly "every previously-classified FB reference". All four
  models are re-run over it (each ref fetches its embedding once, shared cache).
- "No genetic data" comes from the existing ATP:0000207 manual_indexing rows.

``--dry-run`` reports the counts and writes nothing.
"""

import argparse
import logging
import os
import sys

import joblib
import psycopg2

from utils.abc_utils import (download_abc_model, get_model_data, get_tet_source_id,
                             send_classification_tag_to_abc, set_blue_api_base_url)
from agr_document_classifier.agr_document_classifier_classify import (
    classify_documents_from_abc_embeddings, get_confidence_level, configure_logging)

logger = logging.getLogger(__name__)

MOD = "FB"
CLASSIFIER_SOURCE_METHOD = "abc_document_classifier"
CLASSIFIER_SOURCE_DESCRIPTION = ("Alliance document classification pipeline using machine learning "
                                 "to identify papers of interest for curation data types")
# FB per-data-type "reference classification complete" workflow tags. Their union is
# every reference that completed FB reference classification (see module docstring);
# verified superset of new-allele completes, overall-complete and in-progress refs.
REFERENCE_CLASSIFICATION_COMPLETE_TAGS = ["ATP:0000167", "ATP:0000244", "ATP:0000307"]
# The four topics whose classification result is a topic_entity_tag.
TET_TOPICS = {
    "ATP:0000152": "disease",
    "ATP:0000069": "physical interaction",
    "ATP:0000006": "new allele",
    "ATP:0000013": "new transgenic construct",
}
# "No genetic data" — result lives in manual_indexing_tag, not TET.
NO_GEN_DATA_TOPIC = "ATP:0000207"


def _db_params():
    return {
        "dbname": os.getenv("DB_NAME", "literature"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
    }


def get_reference_classification_complete_curies(mod_abbreviation):
    """Curies of references that completed FB reference classification (union of the
    per-data-type 'classification complete' workflow tags)."""
    conn = psycopg2.connect(**_db_params())
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT r.curie "
            "FROM workflow_tag w "
            "JOIN reference r ON w.reference_id = r.reference_id "
            "JOIN mod ON w.mod_id = mod.mod_id "
            "WHERE mod.abbreviation = %s AND w.workflow_tag_id = ANY(%s)",
            (mod_abbreviation, REFERENCE_CLASSIFICATION_COMPLETE_TAGS))
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def get_curies_with_manual_indexing_tag(mod_abbreviation, curation_tag):
    """Curies of references that already have a manual_indexing_tag for ``curation_tag``."""
    conn = psycopg2.connect(**_db_params())
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT r.curie "
            "FROM manual_indexing_tag m "
            "JOIN reference r ON m.reference_id = r.reference_id "
            "JOIN mod ON m.mod_id = mod.mod_id "
            "WHERE mod.abbreviation = %s AND m.curation_tag = %s",
            (mod_abbreviation, curation_tag))
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def update_manual_indexing_confidence(curie_to_confidence, mod_abbreviation, curation_tag):
    """UPDATE manual_indexing_tag.confidence_score (the prediction column) for the
    given references. validation_by_biocurator is never written, so curator
    verdicts are preserved. Returns the number of rows updated."""
    if not curie_to_confidence:
        return 0
    conn = psycopg2.connect(**_db_params())
    updated = 0
    try:
        cur = conn.cursor()
        cur.execute("SELECT mod_id FROM mod WHERE abbreviation = %s", (mod_abbreviation,))
        mod_id = cur.fetchone()[0]
        for curie, confidence in curie_to_confidence.items():
            cur.execute(
                "UPDATE manual_indexing_tag AS m "
                "SET confidence_score = %s, date_updated = NOW() "
                "FROM reference r "
                "WHERE m.reference_id = r.reference_id AND r.curie = %s "
                "AND m.mod_id = %s AND m.curation_tag = %s",
                (float(confidence), curie, mod_id, curation_tag))
            updated += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return updated


def _load_model(topic):
    """Download the current (stage) model for a topic and return (classifier, metadata)."""
    path = f"/data/agr_document_classifier/reclassify_{MOD}_{topic.replace(':', '_')}.joblib"
    download_abc_model(mod_abbreviation=MOD, topic=topic, output_path=path,
                       task_type="biocuration_topic_classification")
    meta = get_model_data(mod_abbreviation=MOD, task_type="biocuration_topic_classification", topic=topic)
    if not meta.get("embedding_profile"):
        raise RuntimeError(f"Model for {topic} is not an ABC-embedding model "
                           f"(embedding_profile is empty); refusing to reclassify with it.")
    return joblib.load(path), meta


def reclassify_tet_topic(topic, name, curies, tet_source_id, cache, dry_run):
    classifier, meta = _load_model(topic)
    logger.info(f"[{topic} {name}] classifying {len(curies)} references")
    species = meta["species"] if (meta.get("species") or "").startswith("NCBITaxon:") else None
    ids, classifications, confidences, valid = classify_documents_from_abc_embeddings(
        curies, MOD, classifier, use_bow=True, embedding_cache=cache)
    sent = skipped = 0
    for curie, cls, conf, ok in zip(ids, classifications, confidences, valid):
        if not ok:
            skipped += 1
            continue
        if dry_run:
            sent += 1
            continue
        ok_sent = send_classification_tag_to_abc(
            curie, species, topic, negated=bool(cls == 0), data_novelty=meta["data_novelty"],
            confidence_score=conf, confidence_level=get_confidence_level(cls, conf),
            tet_source_id=tet_source_id, ml_model_id=meta["ml_model_id"])
        sent += 1 if ok_sent else 0
    logger.info(f"[{topic} {name}] {'would create' if dry_run else 'created'} {sent} TETs "
                f"({skipped} skipped: no embedding)")
    return {"topic": topic, "name": name, "refs": len(curies), "tets": sent, "skipped": skipped}


def reclassify_no_gen_data(cache, dry_run, limit, curies_override=None):
    classifier, meta = _load_model(NO_GEN_DATA_TOPIC)
    if curies_override:
        curies = curies_override
    else:
        curies = get_curies_with_manual_indexing_tag(MOD, NO_GEN_DATA_TOPIC)
        if limit:
            curies = curies[:limit]
    logger.info(f"[{NO_GEN_DATA_TOPIC} no genetic data] {len(curies)} references with an existing tag")
    ids, classifications, confidences, valid = classify_documents_from_abc_embeddings(
        curies, MOD, classifier, use_bow=True, embedding_cache=cache)
    curie_to_conf = {curie: float(conf) for curie, conf, ok in zip(ids, confidences, valid) if ok}
    skipped = len(ids) - len(curie_to_conf)
    if dry_run:
        updated = len(curie_to_conf)
        logger.info(f"[{NO_GEN_DATA_TOPIC}] would update {updated} manual_indexing_tag "
                    f"confidence_score values ({skipped} skipped: no embedding)")
    else:
        updated = update_manual_indexing_confidence(curie_to_conf, MOD, NO_GEN_DATA_TOPIC)
        logger.info(f"[{NO_GEN_DATA_TOPIC}] updated {updated} manual_indexing_tag "
                    f"confidence_score values ({skipped} skipped: no embedding)")
    return {"topic": NO_GEN_DATA_TOPIC, "name": "no genetic data",
            "refs": len(curies), "updated": updated, "skipped": skipped}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Re-classify previously-classified FB references "
                                                 "with the retrained ABC-embedding models "
                                                 "(no workflow/job changes).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts and write nothing.")
    parser.add_argument("--stage", action="store_true",
                        help="Target the stage ABC (where the retrained models live).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap references per topic (smoke test).")
    parser.add_argument("--curies", type=str, default=None,
                        help="Comma-separated reference curies to target instead of enumerating "
                             "(stage verification on a few known-embedded references).")
    parser.add_argument("--skip-no-gen-data", action="store_true",
                        help="Skip the ATP:0000207 manual_indexing update.")
    parser.add_argument("-l", "--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main():
    args = parse_arguments()
    configure_logging(args.log_level)
    if args.stage:
        set_blue_api_base_url("https://stage-literature-rest.alliancegenome.org")
        os.environ["ABC_API_SERVER"] = "https://stage-literature-rest.alliancegenome.org"
        os.environ["ON_PRODUCTION"] = "no"
    if args.dry_run:
        logger.info("DRY RUN: no TETs created, no manual_indexing rows updated.")

    explicit = [c.strip() for c in args.curies.split(",") if c.strip()] if args.curies else None

    tet_source_id = get_tet_source_id(mod_abbreviation=MOD, source_method=CLASSIFIER_SOURCE_METHOD,
                                      source_description=CLASSIFIER_SOURCE_DESCRIPTION)
    if explicit:
        tet_curies = explicit
        logger.info(f"targeting {len(tet_curies)} explicit references (verification run)")
    else:
        tet_curies = get_reference_classification_complete_curies(MOD)
        if args.limit:
            tet_curies = tet_curies[:args.limit]
        logger.info(f"{len(tet_curies)} references completed FB reference classification "
                    f"(shared by the {len(TET_TOPICS)} TET topics)")

    # One embedding cache shared across every model so each reference is fetched once.
    cache: dict = {}
    summary = []
    for topic, name in TET_TOPICS.items():
        summary.append(reclassify_tet_topic(topic, name, tet_curies, tet_source_id, cache, args.dry_run))
    if not args.skip_no_gen_data:
        summary.append(reclassify_no_gen_data(cache, args.dry_run, args.limit, curies_override=explicit))

    logger.info("=" * 60)
    logger.info("Re-classification summary%s:", " (DRY RUN)" if args.dry_run else "")
    for s in summary:
        detail = f"tets={s['tets']}" if "tets" in s else f"manual_indexing_updated={s['updated']}"
        logger.info(f"  {s['topic']} {s['name']:24s} refs={s['refs']:6d} {detail} skipped={s['skipped']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
