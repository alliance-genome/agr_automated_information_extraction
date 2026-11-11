"""Associate genes with references through GenBank/RefSeq accessions (SCRUM-6371).

MOD-agnostic pipeline for references that carry a gene "entity extraction needed"
job:

1. Find GenBank and RefSeq accessions in the main text and each supplement
   (utils.genbank_utils.find_accessions).
2. Link each accession to NCBI Gene IDs with NCBI E-utilities, cached on disk.
3. Map the NCBI Gene IDs to the job MOD's Alliance genes through the A-team
   NCBI_Gene cross-references.
4. Send one gene entity tag per gene with source ``abc_genbank_extraction``
   (ECO:0008023, regular expression method).

The gene extraction jobs are shared with agr_entity_extraction_pipeline_fast.py,
which owns their status, so this pipeline never marks a job started, done or
failed. Run it before the gene extraction pipeline, which completes the jobs.
It sends no negative tags: a paper without accessions says nothing about genes.
"""
import argparse
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TextIO, Tuple

from utils.abc_utils import (
    default_data_context_for_mod,
    download_md_files_for_references,
    get_cached_mod_abbreviation_from_id,
    get_tet_source_id,
    load_all_jobs,
    send_entity_tag_to_abc,
)
from utils.ateam_utils import MOD_TAXON_MAPPING, get_alliance_genes_for_ncbi_gene_ids
from utils.genbank_utils import AccessionHit, NcbiAccessionResolver, find_accessions
from utils.md_utils import AllianceMarkdown

logger = logging.getLogger(__name__)

GENE_TOPIC = "ATP:0000005"
JOB_LABEL = "_extraction_job"
SOURCE_METHOD = "abc_genbank_extraction"
SOURCE_DESCRIPTION = ("Alliance extraction pipeline that uses a regular expression to find GenBank and "
                      "RefSeq accessions in references, links them to NCBI Gene IDs and associates the "
                      "matching Alliance genes with the reference")
DATA_NOVELTY = "ATP:0000335"
MAIN_SOURCE = "main"

DOWNLOAD_DIR = "/data/agr_entity_extraction/to_extract_genbank"
ACCESSION_CACHE_PATH = "/data/agr_entity_extraction/cache/genbank_accession_genes.json"

# A supplement with more accessions than this is a data table (expression
# results, whole-genome lists); tagging every gene in it would swamp curators.
DEFAULT_MAX_ACCESSIONS_PER_SUPPLEMENT = 500
# An accession linked to several genes is a clone, contig or chromosome rather
# than a gene's own sequence.
DEFAULT_MAX_GENES_PER_ACCESSION = 1


@dataclass
class ReferenceHits:
    """Accessions found in one reference, with the files each was found in."""
    hits: Dict[str, AccessionHit] = field(default_factory=dict)
    sources: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    skipped_supplements: List[Tuple[str, int]] = field(default_factory=list)


def read_reference_texts(md_dir: str, curie: str) -> List[Tuple[str, str]]:
    """Return [(source, text)] for a reference's main MD and each supplement MD in ``md_dir``.

    ``source`` is "main" or "supp_<N>". Supplements are read one by one so that a hit
    can be traced to its file and a large data table can be skipped on its own.
    """
    base = curie.replace(":", "_")
    texts: List[Tuple[str, str]] = []
    main_path = os.path.join(md_dir, f"{base}.md")
    if os.path.exists(main_path):
        texts.append((MAIN_SOURCE, _md_fulltext(main_path)))
    supp_prefix = f"{base}.supp_"
    for fname in sorted(os.listdir(md_dir)):
        if fname.startswith(supp_prefix) and fname.endswith(".md"):
            texts.append((fname[len(base) + 1:-len(".md")], _md_fulltext(os.path.join(md_dir, fname))))
    return texts


def _md_fulltext(path: str) -> str:
    md = AllianceMarkdown()
    md.load_from_file(path)
    return md.get_fulltext() or ""


def collect_reference_hits(texts: List[Tuple[str, str]],
                           max_accessions_per_supplement: int = DEFAULT_MAX_ACCESSIONS_PER_SUPPLEMENT
                           ) -> ReferenceHits:
    ref_hits = ReferenceHits()
    for source, text in texts:
        hits = find_accessions(text)
        if source != MAIN_SOURCE and len(hits) > max_accessions_per_supplement:
            ref_hits.skipped_supplements.append((source, len(hits)))
            continue
        for accession, hit in hits.items():
            merged = ref_hits.hits.get(accession)
            if merged is None:
                ref_hits.hits[accession] = AccessionHit(accession=accession, db=hit.db,
                                                        count=hit.count, has_context=hit.has_context)
            else:
                merged.count += hit.count
                merged.has_context = merged.has_context or hit.has_context
            ref_hits.sources[accession].add(source)
    return ref_hits


def assign_genes(ref_hits: ReferenceHits, accession_genes: Dict[str, List[int]],
                 ncbi_to_alliance: Dict[int, List[str]],
                 max_genes_per_accession: int = DEFAULT_MAX_GENES_PER_ACCESSION) -> Dict[str, List[str]]:
    """Return {Alliance gene curie: [accessions]} for one reference."""
    gene_accessions: Dict[str, List[str]] = defaultdict(list)
    for accession in sorted(ref_hits.hits):
        ncbi_genes = accession_genes.get(accession) or []
        if not ncbi_genes or len(ncbi_genes) > max_genes_per_accession:
            continue
        for ncbi_gene in ncbi_genes:
            for gene_curie in ncbi_to_alliance.get(ncbi_gene, []):
                gene_accessions[gene_curie].append(accession)
    return dict(gene_accessions)


def format_note(accessions: List[str], sources: Dict[str, Set[str]]) -> str:
    parts = []
    for accession in accessions:
        found_in = sources.get(accession, set())
        where = [label for label, present in (("main text", MAIN_SOURCE in found_in),
                                              ("supplement", bool(found_in - {MAIN_SOURCE}))) if present]
        parts.append(f"{accession} ({' and '.join(where)})")
    return "GenBank/RefSeq accession: " + ", ".join(parts)


def process_references(mod_abbr: str, curies: List[str], md_dir: str, resolver: NcbiAccessionResolver,
                       test_fh: Optional[TextIO] = None, hits_fh: Optional[TextIO] = None,
                       tet_source_id: Optional[int] = None,
                       max_accessions_per_supplement: int = DEFAULT_MAX_ACCESSIONS_PER_SUPPLEMENT,
                       max_genes_per_accession: int = DEFAULT_MAX_GENES_PER_ACCESSION) -> int:
    """Extract and tag genes for references whose MD files are in ``md_dir``.

    Returns the number of gene tags sent (or written, in test mode).
    """
    taxon = MOD_TAXON_MAPPING[mod_abbr]
    per_reference: Dict[str, ReferenceHits] = {}
    for curie in curies:
        try:
            texts = read_reference_texts(md_dir, curie)
        except Exception as exc:
            logger.warning("%s: could not read MD files: %s", curie, exc)
            continue
        if not texts:
            continue
        ref_hits = collect_reference_hits(texts, max_accessions_per_supplement)
        for source, n_hits in ref_hits.skipped_supplements:
            logger.info("%s: skipping %s with %d accessions (data table)", curie, source, n_hits)
        if ref_hits.hits:
            per_reference[curie] = ref_hits
    if not per_reference:
        return 0

    all_hits: Dict[str, AccessionHit] = {}
    for ref_hits in per_reference.values():
        all_hits.update(ref_hits.hits)
    accession_genes = resolver.resolve(all_hits.values())
    resolver.save_cache()

    ncbi_gene_ids = {gene_id for genes in accession_genes.values()
                     if 0 < len(genes) <= max_genes_per_accession for gene_id in genes}
    ncbi_to_alliance = get_alliance_genes_for_ncbi_gene_ids(ncbi_gene_ids, taxon_curie=taxon)

    n_tags = 0
    data_context = default_data_context_for_mod(mod_abbr, has_entity=True)
    for curie, ref_hits in per_reference.items():
        if hits_fh:
            for accession, hit in sorted(ref_hits.hits.items()):
                ncbi_genes = accession_genes.get(accession) or []
                alliance = sorted({g for n in ncbi_genes for g in ncbi_to_alliance.get(n, [])})
                hits_fh.write("\t".join([
                    curie, ",".join(sorted(ref_hits.sources[accession])), accession, hit.db, str(hit.count),
                    str(int(hit.has_context)), ",".join(map(str, ncbi_genes)), ",".join(alliance)]) + "\n")
        gene_accessions = assign_genes(ref_hits, accession_genes, ncbi_to_alliance, max_genes_per_accession)
        for gene_curie, accessions in sorted(gene_accessions.items()):
            note = format_note(accessions, ref_hits.sources)
            if test_fh:
                test_fh.write(f"{curie}\t{gene_curie}\t{note}\n")
                n_tags += 1
                continue
            ok = send_entity_tag_to_abc(
                reference_curie=curie,
                species=taxon,
                topic=GENE_TOPIC,
                entity_type=GENE_TOPIC,
                entity=gene_curie,
                tet_source_id=tet_source_id,
                data_novelty=DATA_NOVELTY,
                data_context=data_context,
                note=note,
            )
            n_tags += int(bool(ok))
        logger.info("%s: %d accessions, %d genes %s", curie, len(ref_hits.hits), len(gene_accessions),
                    sorted(gene_accessions))
    if test_fh:
        test_fh.flush()
    if hits_fh:
        hits_fh.flush()
    return n_tags


def process_mod(mod_abbr: str, curies: List[str], resolver: NcbiAccessionResolver, batch_size: int,
                test_fh: Optional[TextIO] = None, hits_fh: Optional[TextIO] = None, md_dir: Optional[str] = None,
                **limits) -> None:
    if mod_abbr not in MOD_TAXON_MAPPING:
        logger.warning("No taxon for MOD %s; skipping %d references", mod_abbr, len(curies))
        return
    tet_source_id = None
    if not test_fh:
        tet_source_id = get_tet_source_id(mod_abbreviation=mod_abbr, source_method=SOURCE_METHOD,
                                          source_description=SOURCE_DESCRIPTION)
    if md_dir:
        n_tags = process_references(mod_abbr, curies, md_dir, resolver, test_fh, hits_fh, tet_source_id,
                                    **limits)
        logger.info("%s: %d gene tags from %d references in %s", mod_abbr, n_tags, len(curies), md_dir)
        return
    for start in range(0, len(curies), batch_size):
        batch = curies[start:start + batch_size]
        logger.info("%s: batch of %d references (%d remaining)", mod_abbr, len(batch),
                    len(curies) - start - len(batch))
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        for fname in os.listdir(DOWNLOAD_DIR):
            os.remove(os.path.join(DOWNLOAD_DIR, fname))
        download_md_files_for_references(batch, DOWNLOAD_DIR, mod_abbr, include_supplements=True)
        n_tags = process_references(mod_abbr, batch, DOWNLOAD_DIR, resolver, test_fh, hits_fh, tet_source_id,
                                    **limits)
        logger.info("%s: %d gene tags from this batch", mod_abbr, n_tags)


def curies_in_md_dir(md_dir: str) -> List[str]:
    return sorted(fname[:-len(".md")].replace("_", ":") for fname in os.listdir(md_dir)
                  if fname.endswith(".md") and ".supp_" not in fname)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Associate genes with references via GenBank/RefSeq accessions")
    parser.add_argument("-m", "--mod", action="append",
                        help="Only process these MOD abbreviations (e.g. WB, ZFIN). Repeatable.")
    parser.add_argument("-r", "--reference-curie", action="append",
                        help="Process these references instead of the job queue (needs exactly one --mod). "
                             "Repeatable or comma-separated.")
    parser.add_argument("--md-dir", metavar="DIR",
                        help="Read <curie>.md and <curie>.supp_<N>.md files from DIR instead of downloading "
                             "(needs exactly one --mod).")
    parser.add_argument("-t", "--test-output", metavar="PATH",
                        help="Write '<curie>\\t<gene>\\t<note>' to PATH and send no tags to ABC.")
    parser.add_argument("--hits-output", metavar="PATH",
                        help="Write every accession hit (curie, sources, accession, db, count, has_context, "
                             "NCBI genes, Alliance genes) to PATH for analysis.")
    parser.add_argument("--cache", default=ACCESSION_CACHE_PATH,
                        help=f"Accession -> NCBI Gene cache file (default: {ACCESSION_CACHE_PATH}).")
    parser.add_argument("--max-accessions-per-supplement", type=int, default=DEFAULT_MAX_ACCESSIONS_PER_SUPPLEMENT)
    parser.add_argument("--max-genes-per-accession", type=int, default=DEFAULT_MAX_GENES_PER_ACCESSION)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CLASSIFICATION_BATCH_SIZE", 1000)))
    parser.add_argument("-l", "--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def mod_curies_to_process(args: argparse.Namespace) -> Dict[str, List[str]]:
    wanted_mods = {m.upper() for m in args.mod} if args.mod else None
    if args.reference_curie or args.md_dir:
        if not wanted_mods or len(wanted_mods) != 1:
            raise SystemExit("--reference-curie and --md-dir need exactly one --mod")
        mod_abbr = next(iter(wanted_mods))
        if args.reference_curie:
            curies = [c.strip() for value in args.reference_curie for c in value.split(",") if c.strip()]
        else:
            curies = curies_in_md_dir(args.md_dir)
        return {mod_abbr: curies}

    mod_curies: Dict[str, List[str]] = defaultdict(list)
    for (mod_id, topic), jobs in load_all_jobs(JOB_LABEL, args=None).items():
        if topic != GENE_TOPIC:
            continue
        mod_abbr = get_cached_mod_abbreviation_from_id(mod_id).upper()
        if wanted_mods and mod_abbr not in wanted_mods:
            continue
        mod_curies[mod_abbr].extend(job["reference_curie"] for job in jobs)
    return {mod_abbr: sorted(set(curies)) for mod_abbr, curies in mod_curies.items()}


def main():
    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
    )
    mod_curies = mod_curies_to_process(args)
    if not mod_curies:
        logger.warning("No gene extraction jobs matched; exiting.")
        return

    resolver = NcbiAccessionResolver(cache_path=args.cache)
    limits = {"max_accessions_per_supplement": args.max_accessions_per_supplement,
              "max_genes_per_accession": args.max_genes_per_accession}
    test_fh = open(args.test_output, "w", encoding="utf-8") if args.test_output else None
    hits_fh = open(args.hits_output, "w", encoding="utf-8") if args.hits_output else None
    try:
        for mod_abbr, curies in sorted(mod_curies.items()):
            logger.info("%s: %d references", mod_abbr, len(curies))
            process_mod(mod_abbr, curies, resolver, args.batch_size, test_fh=test_fh, hits_fh=hits_fh,
                        md_dir=args.md_dir, **limits)
    finally:
        for fh in (test_fh, hits_fh):
            if fh:
                fh.close()
    logger.info("Finished GenBank/RefSeq gene extraction.")


if __name__ == "__main__":
    main()
