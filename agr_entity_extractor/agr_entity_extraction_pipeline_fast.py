#!/usr/bin/env python3
"""
Speedups:
- Cache curated entities (names+CURIEs) per (MOD, entity_type) on disk & RAM.
- Pre-filter fulltext to only likely-entity sentences before NER.
- Batch NER safely; fallback if tokenizer lacks pad_token_id.
- Log timing/progress during each batch.
- Skip/quiet HuggingFace pipeline for unsupported custom models.

CLI:
    --tune-threshold          Tune TF-IDF threshold (slow)
    -t PATH / --test-output   Write "<curie>\t<entities>" to PATH instead of sending tags
    -T CURIE --topic CURIE    Filter topics (repeatable) eg -T ATP:0000027
    -m MOD   --mod MOD        Filter MODs (repeatable) eg -m WB
    --ner-batch INT           HF NER batch size (default 16)
    --no-prefilter            Disable regex/dictionary prefilter (slower)
    --log-every INT           Log progress every N papers (default 10)
"""
import argparse
import logging
import os
import re
import socket
import sys
import time
import traceback
import copy
from typing import List
import dill
import requests
from transformers.utils.logging import set_verbosity_error

from utils.abc_utils import (
    load_all_jobs,
    get_cached_mod_abbreviation_from_id,
    get_cached_mod_id_from_abbreviation,
    get_tet_source_id,
    download_abc_model,
    download_md_files_for_references,
    set_job_started,
    set_job_success,
    send_entity_tag_to_abc,
    get_model_data,
    set_job_failure,
)
from utils.ateam_utils import get_all_curated_entities
from utils.md_utils import AllianceMarkdown
from utils.slack_utils import send_slack_notification, format_traceback_html, build_entity_run_summary_html

from utils.entity_extraction_utils import (
    prime_model_entities as prime_model_entities_shared,
    get_model,
    get_pipe,
    run_ner_batched,
    prefilter_text as prefilter_text_generic,
    build_entities_from_results as build_entities_from_results_generic,
    resolve_entity_curie,
    has_allele_like_context,
    rescue_short_alleles_from_fulltext,
    filter_false_positive_alleles,
    is_allele_topic,
    ABC_ALLELE_TOPIC,
    SUSPICIOUS_PREFIX_RE,
    ALLELE_NAME_PATTERN,
    GENERIC_NAME_PATTERN,
    STRAIN_NAME_PATTERN,
    get_target_entities,
    apply_tfidf_count_gate,
    ZFIN_GENE_ALLELE_FALSE_POSITIVE_WORDS,
)

# Silence HF info/warnings entirely
set_verbosity_error()

logger = logging.getLogger(__name__)

TOPIC2TYPE = {
    "ATP:0000027": "strain",
    "ATP:0000110": "transgenic_allele",
    "ATP:0000006": "allele",
    "ATP:0000285": "allele",   # classical allele (curated entity_type is still "allele")
    "ATP:0000123": "species",  # not used here but mapped for completeness
    "ATP:0000005": "gene",
}

# Pandoc preserves gene italics as ``*symbol*`` in the article markdown.  This
# gives all-letter ZFIN symbols a much stronger signal than an unrestricted
# body-text dictionary match, which also matches thousands of ordinary words.
MARKDOWN_ITALIC_SPAN_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
MARKDOWN_GENE_LIST_ITEM_RE = re.compile(r"[A-Za-z0-9_.\-]+")
MARKDOWN_REFERENCES_RE = re.compile(
    r"^#{1,3}\s+(?:references|bibliography|literature cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def topic_to_entity_type(topic: str) -> str:
    return TOPIC2TYPE.get(topic, "gene")


def get_generic_name_pattern(topic: str) -> re.Pattern:
    if topic == "ATP:0000027":       # strain
        return STRAIN_NAME_PATTERN
    if is_allele_topic(topic):       # allele / classical allele
        return ALLELE_NAME_PATTERN
    return GENERIC_NAME_PATTERN      # gene, transgene, etc.


# --------------------------------------------------------------------- #
# Priming: load curated entities into the model                         #
# --------------------------------------------------------------------- #
def prime_model_entities(model, mod_abbr: str, topic: str):
    entity_type = topic_to_entity_type(topic)
    prime_model_entities_shared(
        model=model,
        mod_abbr=mod_abbr,
        entity_type=entity_type,
        loader_fn=get_all_curated_entities,
    )
    # Keep the MOD on the model so MOD-specific precision rules can be applied
    # after the shared curated-list priming step.
    model.mod_abbr = mod_abbr


def rescue_zfin_all_letter_genes_from_markdown(fulltext: str, model) -> List[str]:
    """Return curated all-letter ZFIN symbols presented as italic gene names.

    A plain full-body dictionary scan has poor precision because many curated
    symbols are also English/lab words.  Gene typography is a scalable signal:
    accept a single italic token, or tokens in an italic comma/semicolon list,
    and always intersect them with the current cached curated list.
    """
    if (
        not fulltext
        or getattr(model, "mod_abbr", None) != "ZFIN"
        or getattr(model, "topic", None) != "ATP:0000005"
    ):
        return []

    upper_to_original = getattr(model, "upper_to_original_mapping", {}) or {}
    rescued: set[str] = set()
    references = MARKDOWN_REFERENCES_RE.search(fulltext)
    article_body = fulltext[:references.start()] if references else fulltext
    for span_match in MARKDOWN_ITALIC_SPAN_RE.finditer(article_body):
        span = span_match.group(1)
        # Long spans usually come from unmatched markdown asterisks. Colons,
        # parentheses, and HTML tags identify constructs/genotypes rather than
        # standalone gene-symbol typography.
        if len(span) > 200 or re.search(r"[:()<>]", span):
            continue
        # A parenthesized italic word such as ``(*top*)`` is normally a figure
        # orientation/label, not a gene mention.
        if (
            span_match.start() > 0
            and span_match.end() < len(article_body)
            and article_body[span_match.start() - 1] == "("
            and article_body[span_match.end()] == ")"
        ):
            continue
        # Accept complete italic symbols and complete comma/semicolon list
        # items. Do not mine substrings from constructs such as ``*5’del*`` or
        # partially emphasized symbols such as ``*spi1b*``.
        for item in re.split(r"[,;]", span):
            token = item.strip().rstrip(".")
            if not MARKDOWN_GENE_LIST_ITEM_RE.fullmatch(token):
                continue
            curated = upper_to_original.get(token.upper())
            # ZFIN gene symbols are lower-case. Exact case avoids treating
            # upper-case human proteins/antibodies as zebrafish genes.
            if curated == token and curated.isalpha():
                rescued.add(curated)
    return sorted(rescued)


# Genotype nomenclature renders the allele as a superscript on its gene, e.g.
# ``nkx3.1<sup>ca116</sup>`` or ``sdhb<sup>rmc200</sup>``. Once the markup is
# flattened for NER the tags vanish and the allele fuses onto the gene token
# (``nkx3.1ca116``, ``sdhbrmc200``), so ALLELE_NAME_PATTERN — which requires a
# non-alphanumeric left delimiter — can no longer isolate it. Recover the allele
# straight from the superscript span in the raw markdown instead.
MARKDOWN_SUPERSCRIPT_SPAN_RE = re.compile(r"<sup>(.*?)</sup>", re.IGNORECASE | re.DOTALL)
MARKDOWN_EMPHASIS_CHARS_RE = re.compile(r"[*_`\s]+")


def rescue_superscript_alleles_from_markdown(raw_markdown: str, model) -> List[str]:
    """Return curated alleles written as a gene superscript (``gene<sup>allele</sup>``).

    Such alleles fuse onto the preceding gene once the markup is stripped, so the
    standalone-token scan misses them. Intersecting each superscript's cleaned
    content with the curated allele list keeps this safe: only an exact curated
    allele name (case included) can ever be rescued, so citation/zygosity
    superscripts (``<sup>1,2</sup>``, ``<sup>+/-</sup>``) are ignored.
    """
    if not raw_markdown:
        return []
    curated = getattr(model, "entities_to_extract", None)
    if not curated:
        return []
    curated_set = {c for c in curated if isinstance(c, str)}
    if not curated_set:
        return []
    rescued: set[str] = set()
    for span_match in MARKDOWN_SUPERSCRIPT_SPAN_RE.finditer(raw_markdown):
        # Strip markdown emphasis markers and whitespace the pandoc output leaves
        # inside the span (e.g. ``*ca116*``) before matching.
        inner = MARKDOWN_EMPHASIS_CHARS_RE.sub("", span_match.group(1) or "")
        if not inner:
            continue
        # A superscript may hold a bare allele (``ca116``) or one decorated with a
        # zygosity/reference marker (``ca116/+``); test the whole cleaned token and
        # each slash-delimited part.
        for token in [inner, *inner.split("/")]:
            if token and token in curated_set:
                rescued.add(token)
    return sorted(rescued)


# --------------------------------------------------------------------- #
# Prefilter + Build helpers (via shared utils)                          #
# --------------------------------------------------------------------- #
def prefilter_text(fulltext: str, model) -> str:
    return prefilter_text_generic(
        fulltext=fulltext,
        model=model,
        pattern=get_generic_name_pattern(model.topic),
        is_species=False,               # no species-specific normalization
        normalize_aliases=None,
        expand_abbrevs=None,
        use_gold_substring=False,
    )


def build_entities_from_results(results, title: str, abstract: str, fulltext: str, model, raw_markdown: str = "") -> List[str]:
    allele_mode = is_allele_topic(getattr(model, "topic", None))

    entities = build_entities_from_results_generic(
        results=results,
        title=title,
        abstract=abstract,
        fulltext=fulltext,
        model=model,
        pattern=get_generic_name_pattern(model.topic),
        is_species=False,
        normalize_aliases=None,
        expand_abbrevs=None,
        use_fulltext_tokenizer=False,
        use_count_gate=False,
        # Alleles are matched with strict case-sensitivity so look-alikes that
        # differ only by case (e.g. the balancer chromosome 'qC1' vs. the curated
        # allele 'qc1') are never conflated. Candidates returned here already
        # equal a curated allele name exactly.
        case_sensitive=allele_mode,
    )

    # The generic body regex requires a digit for precision.  Recover no-digit
    # ZFIN genes only when article typography marks them as gene symbols.
    italic_gene_rescues = rescue_zfin_all_letter_genes_from_markdown(raw_markdown or fulltext, model)
    if italic_gene_rescues:
        logger.info(
            "ZFIN-GENE-RESCUE: adding %d curated italic all-letter genes: %s",
            len(italic_gene_rescues), ", ".join(italic_gene_rescues),
        )
        entities = sorted(set(entities) | set(italic_gene_rescues))

    # Alleles written as a gene superscript (gene<sup>allele</sup>) fuse onto the
    # gene when the markup is flattened, so recover them from the raw markdown.
    # High-confidence like the italic gene rescues: re-added after all filtering.
    superscript_allele_rescues = (
        rescue_superscript_alleles_from_markdown(raw_markdown or fulltext, model)
        if allele_mode
        else []
    )

    # For allele topic, enforce allele-specific post-processing rules.
    if allele_mode:
        original_count = len(entities)

        # Candidates already match a curated allele name EXACTLY (case included),
        # so no lowercase normalization / uppercase remapping is applied here —
        # doing so would defeat the strict case-sensitivity the curator asked for.

        # 1) Drop suspicious one-letter+digit alleles (e5, e7, b2, ...) unless context looks allele-like
        before_suspicious_filter = len(entities)
        filtered_entities: list[str] = []
        for e in entities:
            if SUSPICIOUS_PREFIX_RE.match(e):
                if not has_allele_like_context(fulltext, e):
                    logger.info(
                        "ALLELE-FILTER: dropping suspicious short allele '%s' due to weak context",
                        e,
                    )
                    continue
            filtered_entities.append(e)
        entities = filtered_entities
        dropped_suspicious = before_suspicious_filter - len(entities)

        # 2) Rescue allele-like tokens from fulltext that NER missed (case preserved)
        already_found = set(entities)
        rescued = rescue_short_alleles_from_fulltext(fulltext, model, already_found, case_sensitive=True)
        if rescued:
            logger.info(
                "ALLELE-RESCUE: adding %d alleles from fulltext that NER missed: %s",
                len(rescued), ", ".join(sorted(rescued))
            )
            entities.extend(sorted(rescued))

        # 3) Normalize trailing dots and dedupe after rescue
        normalized: list[str] = []
        seen: set[str] = set()
        for e in entities:
            e_norm = e.rstrip('.')
            if e_norm and e_norm not in seen:
                seen.add(e_norm)
                normalized.append(e_norm)
        entities = normalized

        # 4) Strict case-sensitive curated filter: a candidate must match a curated
        #    allele name EXACTLY (case included). This is what keeps balancer 'qC1'
        #    from passing as the curated allele 'qc1' once rescue has added it.
        curated = getattr(model, "entities_to_extract", None)
        dropped_non_curated = 0
        if curated:
            curated_set = {c for c in curated if isinstance(c, str)}
            before_curated_filter = len(entities)
            entities = [e for e in entities if e in curated_set]
            dropped_non_curated = before_curated_filter - len(entities)

        # 5) Context-based false positive filtering
        #    Filter out alleles that are markers, balancers, reagents, etc.
        before_fp_filter = len(entities)
        entities, _ = filter_false_positive_alleles(entities, fulltext)
        dropped_false_positives = before_fp_filter - len(entities)

        if dropped_suspicious > 0 or dropped_non_curated > 0 or dropped_false_positives > 0:
            logger.info(
                "ALLELE-FILTER: start=%d, dropped_suspicious=%d, "
                "dropped_non_curated=%d, dropped_false_positives=%d, final=%d",
                original_count,
                dropped_suspicious,
                dropped_non_curated,
                dropped_false_positives,
                len(entities),
            )

    # Apply the model's min_matches + tf-idf gate. This is a no-op when the
    # model's tfidf_threshold<=0 and min_matches<=1 (e.g. an initial extractor),
    # and the precision filter once a tuned threshold is set. Then drop curated
    # symbols that are common English words (author/methods collisions), which
    # the tf-idf gate misses at threshold 0.
    count_text = "\n".join(t for t in (title, abstract, fulltext) if t)
    before_gate = len(entities)
    entities = apply_tfidf_count_gate(entities, count_text, model)
    # Italic-typography rescues are high-confidence: re-add any the tf-idf/count
    # gate dropped (e.g. single-mention curated genes) so a strong typographic
    # signal is not overridden by a frequency threshold.
    if italic_gene_rescues:
        entities = sorted(set(entities) | set(italic_gene_rescues))
    if superscript_allele_rescues:
        logger.info(
            "ALLELE-SUPERSCRIPT-RESCUE: adding %d curated superscript alleles: %s",
            len(superscript_allele_rescues), ", ".join(superscript_allele_rescues),
        )
        entities = sorted(set(entities) | set(superscript_allele_rescues))
    # The collision stopword list is derived entirely from the ZFIN test set, so it
    # is applied ONLY for ZFIN. Other MODs (WB, etc.) are left untouched.
    if getattr(model, "mod_abbr", None) == "ZFIN":
        entities = [e for e in entities if e.lower() not in ZFIN_GENE_ALLELE_FALSE_POSITIVE_WORDS]
    if len(entities) != before_gate:
        logger.info("GATE/STOPWORD: %d -> %d (tfidf_threshold=%s, min_matches=%s)",
                    before_gate, len(entities),
                    getattr(model, "tfidf_threshold", None), getattr(model, "min_matches", None))

    return entities


# --------------------------------------------------------------------- #
# Threshold tuning                                                      #
# --------------------------------------------------------------------- #
def find_best_tfidf_threshold(mod_id, topic, jobs, target_entities, combined_md_dir=None):
    mod_abbr = get_cached_mod_abbreviation_from_id(mod_id)
    model_fp = f"/data/agr_document_classifier/biocuration_entity_extraction_{mod_abbr}_{topic.replace(':', '_')}.dpkl"
    try:
        download_abc_model(mod_abbreviation=mod_abbr, topic=topic, output_path=model_fp,
                           task_type="biocuration_entity_extraction")
        logger.info("Classification model downloaded for mod=%s, topic=%s.", mod_abbr, topic)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning("Model not found for mod=%s, topic=%s. Skipping.", mod_abbr, topic)
            return None
        raise

    entity_model = dill.load(open(model_fp, "rb"))
    entity_model.alliance_entities_loaded = True
    # get_model() sets this on the normal path; find_best loads the model
    # directly, so set it here. build_entities_from_results needs model.topic.
    entity_model.topic = topic

    best_threshold = 0.1
    best_similarity = -1.0
    thresholds = [i / 10.0 for i in range(1, 51)]

    def score_dir(md_dir):
        """Sweep thresholds over the MD files already present in md_dir, scoring
        Jaccard vs the gold set; updates best_threshold/best_similarity.

        Uses the SAME extraction path as the pipeline (build_entities_from_results,
        which applies the model's tf-idf/min_matches gate + stopword filter), so
        the tuned threshold reflects what extraction actually produces."""
        nonlocal best_threshold, best_similarity
        md_files = [f for f in os.listdir(md_dir) if f.endswith(".md") and ".supp_" not in f]
        # Preload each paper's text once; only the threshold changes per sweep step.
        docs = []
        for f in md_files:
            curie = f.split(".")[0].replace("_", ":")
            try:
                md = AllianceMarkdown()
                md.load_from_file(os.path.join(md_dir, f))
                title = md.get_title() or ""
                abstract = md.get_abstract() or ""
                fulltext = (
                    md.get_fulltext(include_attributes=True)
                    if is_allele_topic(topic)
                    else md.get_fulltext()
                ) or ""
            except Exception as e:
                logger.warning("Error loading MD for %s: %s. Skipping.", curie, e)
                continue
            docs.append((curie, title, abstract, fulltext, md.raw_md))

        for th in thresholds:
            entity_model.tfidf_threshold = th
            total_sim = 0.0
            count = 0
            for curie, title, abstract, fulltext, raw_markdown in docs:
                entities = build_entities_from_results([], title, abstract, fulltext, entity_model, raw_markdown=raw_markdown)
                all_low = {e.lower() for e in entities}
                gold_low = {e.lower() for e in target_entities.get(curie, [])}
                if all_low or gold_low:
                    sim = len(all_low & gold_low) / len(all_low | gold_low)
                    total_sim += sim
                    count += 1

            avg_sim = (total_sim / count) if count else 0.0
            logger.info("Threshold %.1f: Average Jaccard %.4f (%d papers)", th, avg_sim, count)
            if avg_sim > best_similarity:
                best_similarity = avg_sim
                best_threshold = th

    if combined_md_dir:
        # No jobs needed: tune over the MDs already in the directory (e.g. a curated
        # test set for a MOD that has no queued extraction jobs, like ZFIN).
        n = len([f for f in os.listdir(combined_md_dir) if f.endswith(".md")])
        logger.info("Tuning over combined MD dir %s (%d MD files) - no jobs required.", combined_md_dir, n)
        score_dir(combined_md_dir)
    else:
        batch_size = int(os.environ.get("CLASSIFICATION_BATCH_SIZE", 1000))
        jobs_to_process = copy.deepcopy(jobs)
        while jobs_to_process:
            batch = jobs_to_process[:batch_size]
            ref_map = {job["reference_curie"]: job for job in batch}
            jobs_to_process = jobs_to_process[batch_size:]
            logger.info("Processing a batch of %d jobs. Remaining: %d", len(batch), len(jobs_to_process))

            out_dir = "/data/agr_entity_extraction/to_extract"
            os.makedirs(out_dir, exist_ok=True)
            for f in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, f))
            download_md_files_for_references(list(ref_map.keys()), out_dir, mod_abbr)
            score_dir(out_dir)

    logger.info("Best TFIDF threshold: %.1f (Jaccard %.4f)", best_threshold, best_similarity)
    return best_threshold


# --------------------------------------------------------------------- #
# Core processing                                                       #
# --------------------------------------------------------------------- #
def process_entity_extraction_jobs(mod_id, topic, jobs, test_mode: bool = False, test_fh=None, ner_batch_size: int = 16, prefilter: bool = True, log_every: int = 10, combined_md_dir: bool = False):  # noqa: C901
    mod_abbr = get_cached_mod_abbreviation_from_id(mod_id)

    # The extraction job may be keyed on either the generic "allele" topic
    # (ATP:0000006) or the "classical allele" topic (ATP:0000285), but the tag we
    # report to ABC always uses "classical allele" (ATP:0000285) for both topic
    # and entity_type, to match the WB allele data import. Non-allele topics are
    # reported unchanged.
    abc_topic = ABC_ALLELE_TOPIC if is_allele_topic(topic) else topic

    tet_source_id = None
    if not test_mode:
        tet_source_id = get_tet_source_id(
            mod_abbreviation=mod_abbr,
            source_method="abc_entity_extractor",
            source_description=("Alliance entity extraction pipeline using machine learning "
                                "to identify papers of interest for curation data types")
        )

    model_fp = f"/data/agr_document_classifier/biocuration_entity_extraction_{mod_abbr}_{topic.replace(':', '_')}.dpkl"

    try:
        meta = get_model_data(mod_abbreviation=mod_abbr,
                              task_type="biocuration_entity_extraction",
                              topic=topic)
        species = meta["species"]
        data_novelty = meta['data_novelty']
        ml_model_id = meta['ml_model_id']
        download_abc_model(mod_abbreviation=mod_abbr,
                           topic=topic,
                           output_path=model_fp,
                           task_type="biocuration_entity_extraction")
        logger.info("Classification model downloaded for mod=%s, topic=%s", mod_abbr, topic)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning("Model not found for mod=%s, topic=%s. Skipping.", mod_abbr, topic)
            return {"failed": 0, "md_skipped": 0,
                    "skipped": [{"mod_abbreviation": mod_abbr, "topic": topic,
                                 "jobs": len(jobs), "reason": "extraction model not found"}]}
        raise

    model = get_model(mod_abbr, topic, model_fp)
    if not getattr(model, "alliance_entities_loaded", False):
        logger.info("Priming Alliance entity lists for %s/%s", mod_abbr, topic)
        prime_model_entities(model, mod_abbr, topic)

    ner_pipe = get_pipe(mod_abbr, topic, model)

    # ---------------- combined MD dir path ---------------- #
    if combined_md_dir:
        logger.info("Processing all combined MD files in %s (files will not be deleted)", combined_md_dir)
        for fname in sorted(os.listdir(combined_md_dir)):
            if not fname.endswith(".md"):
                continue
            curie = fname.replace(".md", "")
            path = os.path.join(combined_md_dir, fname)
            try:
                md = AllianceMarkdown()
                md.load_from_file(path)
                title = md.get_title() or ""
                abstract = md.get_abstract() or ""
                fulltext = (
                    md.get_fulltext(include_attributes=True)
                    if is_allele_topic(model.topic)
                    else md.get_fulltext()
                ) or ""
            except Exception as e:
                logger.warning("Failed to load/process %s: %s", fname, e)
                continue

            text_for_ner = prefilter_text(fulltext, model) if prefilter else fulltext
            results = run_ner_batched(ner_pipe, [text_for_ner], ner_batch_size)[0]
            all_entities = build_entities_from_results(results, title, abstract, fulltext, model, raw_markdown=md.raw_md)

            if test_mode:
                test_fh.write(f"{curie}\t{' | '.join(all_entities)}\n")
                test_fh.flush()
            else:
                if not all_entities:
                    send_entity_tag_to_abc(
                        reference_curie=curie,
                        species=species,
                        topic=abc_topic,
                        negated=True,
                        tet_source_id=tet_source_id,
                        data_novelty=data_novelty,
                        ml_model_id=ml_model_id
                    )
                else:
                    seen = set()
                    for ent in all_entities:
                        # Allele-specific mapping debug / behavior
                        if is_allele_topic(topic):
                            try:
                                ent_curie = resolve_entity_curie(model, ent, strict=True)
                            except KeyError:
                                logger.info(
                                    "ALLELE-REJECT: mapping KeyError for candidate '%s' in ref %s (combined MD)",
                                    ent, curie,
                                )
                                continue
                            if not ent_curie:
                                logger.info(
                                    "ALLELE-REJECT: no CURIE found for candidate '%s' in ref %s (combined MD)",
                                    ent, curie,
                                )
                                continue
                        else:
                            try:
                                ent_curie = resolve_entity_curie(model, ent, strict=True)
                            except KeyError:
                                raise

                        if not ent_curie or ent_curie in seen:
                            continue
                        seen.add(ent_curie)
                        # For strains, use entity-specific taxon if available
                        if topic == "ATP:0000027":
                            entity_species = getattr(model, "curie_to_taxon_mapping", {}).get(ent_curie, species)
                        else:
                            entity_species = species
                        send_entity_tag_to_abc(
                            reference_curie=curie,
                            species=entity_species,
                            topic=abc_topic,
                            entity_type=abc_topic,
                            entity=ent_curie,
                            tet_source_id=tet_source_id,
                            data_novelty=data_novelty,
                            ml_model_id=ml_model_id
                        )

            logger.info("%s => %s", curie, all_entities)
        logger.info("Finished processing combined MD directory.")
        return {"failed": 0, "md_skipped": 0, "skipped": []}

    # ---------------- job-based processing ---------------- #
    failed_count = 0
    md_skipped = 0
    classification_batch_size = int(os.environ.get("CLASSIFICATION_BATCH_SIZE", 1000))
    jobs_to_process = copy.deepcopy(jobs)

    while jobs_to_process:
        job_batch = jobs_to_process[:classification_batch_size]
        jobs_to_process = jobs_to_process[classification_batch_size:]

        ref_map = {j["reference_curie"]: j for j in job_batch}
        logger.info("Processing batch of %d jobs. Remaining: %d", len(job_batch), len(jobs_to_process))

        out_dir = "/data/agr_entity_extraction/to_extract"
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))

        download_md_files_for_references(list(ref_map.keys()), out_dir, mod_abbr)

        metas: List[tuple[str, dict, str, str, str]] = []  # (curie, job, title, abstract, fulltext)
        texts_for_ner: List[str] = []

        # ---- Prepare MDs ----
        for fname in os.listdir(out_dir):
            curie = fname.split(".")[0].replace("_", ":")
            job = ref_map.get(curie)
            if job is None:
                continue
            try:
                md = AllianceMarkdown()
                md.load_from_file(os.path.join(out_dir, fname))
            except Exception as e:
                logger.warning("MD load failed for %s: %s. Skipping.", curie, e)
                md_skipped += 1
                continue

            try:
                fulltext = (
                    md.get_fulltext(include_attributes=True)
                    if is_allele_topic(model.topic)
                    else md.get_fulltext()
                ) or ""
            except Exception as e:
                logger.error("Fulltext error for %s: %s. Marking failure.", curie, e)
                set_job_started(job)
                set_job_failure(job)
                failed_count += 1
                continue

            try:
                abstract = md.get_abstract() or ""
            except Exception as e:
                logger.warning("Abstract error for %s: %s. Ignoring.", curie, e)
                abstract = ""
            try:
                title = md.get_title() or ""
            except Exception as e:
                logger.warning("Title error for %s: %s. Ignoring.", curie, e)
                title = ""

            text_for_ner = prefilter_text(fulltext, model) if prefilter else fulltext
            texts_for_ner.append(text_for_ner)
            metas.append((curie, job, title, abstract, fulltext, md.raw_md))

        if not texts_for_ner:
            logger.info("No valid MDs in this batch.")
            continue

        # ---- Run NER ----
        t0 = time.perf_counter()
        results_list = run_ner_batched(ner_pipe, texts_for_ner, ner_batch_size)
        total_time = time.perf_counter() - t0
        logger.info("NER on %d docs took %.1fs (%.2fs/doc)", len(texts_for_ner), total_time, total_time / len(texts_for_ner))

        # ---- Post-process ----
        for idx, ((curie, job, title, abstract, fulltext, raw_markdown), results) in enumerate(zip(metas, results_list), 1):
            all_entities = build_entities_from_results(results, title, abstract, fulltext, model, raw_markdown=raw_markdown)

            if test_mode:
                test_fh.write(f"{curie}\t{' | '.join(all_entities)}\n")
                test_fh.flush()
            else:
                if not all_entities:
                    send_entity_tag_to_abc(
                        reference_curie=curie,
                        species=species,
                        topic=abc_topic,
                        negated=True,
                        tet_source_id=tet_source_id,
                        data_novelty=data_novelty,
                        ml_model_id=ml_model_id
                    )
                else:
                    seen = set()
                    for ent in all_entities:
                        # Allele-specific mapping debug / behavior
                        if is_allele_topic(topic):
                            try:
                                ent_curie = resolve_entity_curie(model, ent, strict=True)
                            except KeyError:
                                logger.info(
                                    "ALLELE-REJECT: mapping KeyError for candidate '%s' in ref %s",
                                    ent, curie,
                                )
                                continue
                            if not ent_curie:
                                logger.info(
                                    "ALLELE-REJECT: no CURIE found for candidate '%s' in ref %s",
                                    ent, curie,
                                )
                                continue
                        else:
                            try:
                                ent_curie = resolve_entity_curie(model, ent, strict=True)
                            except KeyError:
                                raise

                        if not ent_curie or ent_curie in seen:
                            continue
                        seen.add(ent_curie)
                        # For strains, use entity-specific taxon if available
                        if topic == "ATP:0000027":
                            entity_species = getattr(model, "curie_to_taxon_mapping", {}).get(ent_curie, species)
                        else:
                            entity_species = species
                        send_entity_tag_to_abc(
                            reference_curie=curie,
                            species=entity_species,
                            topic=abc_topic,
                            entity_type=abc_topic,
                            entity=ent_curie,
                            tet_source_id=tet_source_id,
                            data_novelty=data_novelty,
                            ml_model_id=ml_model_id
                        )

            if idx % log_every == 0:
                logger.info("Processed %d/%d in this batch", idx, len(metas))

            set_job_started(job)
            set_job_success(job)
            logger.info("%s = %s", curie, all_entities)

        logger.info("Finished processing batch of %d jobs.", len(job_batch))
    return {"failed": failed_count, "md_skipped": md_skipped, "skipped": []}


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description='Extract biological entities from documents')
    parser.add_argument(
        "--combined-md-dir", metavar="DIR",
        help="Process every .md file under this directory instead of downloading MDs."
    )
    parser.add_argument("--tune-threshold", action="store_true",
                        help="Run find_best_tfidf_threshold on all jobs (slow, for experimentation only)")
    parser.add_argument("-l", "--log_level", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="Set the logging level")
    parser.add_argument("-t", "--test-output", metavar="PATH",
                        help="Write '<curie>\\t<pipe_separated_entities>' to PATH and skip sending tags to ABC")
    parser.add_argument("-T", "--topic", action="append",
                        help="Only process these topic CURIE(s). Repeatable.")
    parser.add_argument("-m", "--mod", action="append",
                        help="Only process these MOD abbreviations (e.g. WB, ZFIN). Repeatable.")
    parser.add_argument("--ner-batch", type=int, default=16,
                        help="Batch size for the HuggingFace NER pipeline (default: 16).")
    parser.add_argument("--no-prefilter", action="store_true",
                        help="Disable regex/dictionary prefiltering before NER (slower).")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log progress every N papers inside a batch (default: 10).")
    parser.set_defaults(strict_mapping=True)

    args = parser.parse_args()

    # ---------- NEW DEFAULTS WHEN no --mod or no -- topic provided ----------
    DEFAULT_MODS = ['WB']
    # strain, gene, transgene, allele
    DEFAULT_TOPICS = ['ATP:0000027', 'ATP:0000005', 'ATP:0000110', 'ATP:0000006']
    if not args.mod:
        args.mod = DEFAULT_MODS
        logging.getLogger(__name__).info("No --mod provided; defaulting to %s", ", ".join(DEFAULT_MODS))
    if not args.topic:
        args.topic = DEFAULT_TOPICS
        logging.getLogger(__name__).info("No --topic provided; defaulting to %s", ", ".join(DEFAULT_TOPICS))
    # ------------------------------------------------------------------------

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )

    _mod_cache = {}

    def mod_id_to_abbr(mod_id):
        if mod_id not in _mod_cache:
            _mod_cache[mod_id] = get_cached_mod_abbreviation_from_id(mod_id).upper()
        return _mod_cache[mod_id]

    if args.combined_md_dir:
        # Direct mode: process/tune over the MD directory for the requested
        # MOD(s)/topic(s) without requiring queued extraction jobs. This makes
        # --combined-md-dir (and --tune-threshold with it) usable for a MOD that
        # has no extraction jobs yet (e.g. ZFIN). MDs are read from disk, so the
        # job lists are irrelevant and left empty.
        mod_topic_jobs = {
            (get_cached_mod_id_from_abbreviation(m.upper()), topic): []
            for m in args.mod
            for topic in args.topic
        }
        logger.info("combined-md-dir mode: %s for MOD(s)=%s topic(s)=%s",
                    args.combined_md_dir, ", ".join(args.mod), ", ".join(args.topic))
    else:
        mod_topic_jobs = load_all_jobs("_extraction_job", args=None)

        wanted_topics = set(args.topic) if args.topic else None
        wanted_mods = {m.upper() for m in (args.mod or [])} if args.mod else None

        filtered = {}
        for (mod_id, topic), jobs in mod_topic_jobs.items():
            if wanted_topics and topic not in wanted_topics:
                continue
            if wanted_mods and mod_id_to_abbr(mod_id) not in wanted_mods:
                continue
            filtered[(mod_id, topic)] = jobs
        mod_topic_jobs = filtered

        if not mod_topic_jobs:
            logger.warning("No jobs matched the provided filters (topic/mod). Exiting.")
            return

    if args.tune_threshold:
        for (mod_id, topic), jobs in mod_topic_jobs.items():
            mod_abbr = mod_id_to_abbr(mod_id)
            TARGET = get_target_entities(mod_abbr, topic)
            best = find_best_tfidf_threshold(mod_id, topic, jobs, TARGET,
                                             combined_md_dir=args.combined_md_dir)
            logger.info("Best TF-IDF threshold for %s/%s: %s", mod_id, topic, best)
        logger.info("Threshold tuning complete.")
        return

    test_mode = bool(args.test_output)
    test_fh = open(args.test_output, "w", encoding="utf-8") if test_mode else None
    host = socket.gethostname()
    total_jobs = sum(len(jobs) for jobs in mod_topic_jobs.values())
    total_failed = 0
    total_md_skipped = 0
    skipped_jobs = []
    try:
        for (mod_id, topic), jobs in mod_topic_jobs.items():
            result = process_entity_extraction_jobs(
                mod_id,
                topic,
                jobs,
                test_mode=test_mode,
                test_fh=test_fh,
                ner_batch_size=args.ner_batch,
                prefilter=not args.no_prefilter,
                log_every=args.log_every,
                combined_md_dir=args.combined_md_dir,
            )
            total_failed += result["failed"]
            total_md_skipped += result["md_skipped"]
            skipped_jobs.extend(result["skipped"])
    except Exception:
        send_slack_notification(
            f":x: Entity extraction job CRASHED on {host} (PRODUCTION)",
            format_traceback_html(traceback.format_exc())
        )
        raise
    finally:
        if test_fh:
            test_fh.close()

    logger.info("Finished processing all entity extraction jobs.")
    if (total_failed or total_md_skipped or skipped_jobs) and not test_mode:
        send_slack_notification(
            f":warning: Entity extraction finished with issues on {host} (PRODUCTION)",
            build_entity_run_summary_html(total_failed, total_jobs, total_md_skipped, skipped_jobs)
        )


if __name__ == '__main__':
    main()
