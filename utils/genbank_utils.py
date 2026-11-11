"""Find GenBank/RefSeq accessions in text and resolve them to NCBI Gene IDs.

Used by the MOD-agnostic GenBank extraction pipeline (SCRUM-6371): accessions
found in a paper's full text and supplements are linked to NCBI Gene IDs via
NCBI E-utilities, and the pipeline then maps those IDs to Alliance genes through
the A-team NCBI_Gene cross-references.

Accession formats follow NCBI's documentation
(https://www.ncbi.nlm.nih.gov/genbank/acc_prefix/ and the RefSeq prefix table).
The GenBank formats are short letter+digit strings that also match UniProt
accessions, GEO/SRA identifiers, WormBase non-elegans gene IDs, grant numbers and
catalog numbers, so a regex hit is only a candidate: it becomes a gene only when
NCBI links it to exactly one gene and that gene is an Alliance gene for the MOD.
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)

# RefSeq: a two-letter prefix, an underscore and 6 or 9 digits. Only the
# gene-level prefixes are kept; NC_/NT_/NW_/AC_ (chromosomes, contigs) link to
# many genes and WP_ (non-redundant prokaryotic protein) has no gene.
REFSEQ_NUCLEOTIDE_PREFIXES = ("NM", "NR", "XM", "XR", "NG")
REFSEQ_PROTEIN_PREFIXES = ("NP", "XP", "YP", "AP")
REFSEQ_RE = re.compile(
    r'(?<![A-Za-z0-9])'
    r'((?:' + '|'.join(REFSEQ_NUCLEOTIDE_PREFIXES + REFSEQ_PROTEIN_PREFIXES) + r')_\d{6,9})'
    r'(?:\.\d+)?'
    r'(?![A-Za-z0-9])'
)

# INSDC (GenBank/ENA/DDBJ). Boundaries allow "_" because the converted Markdown
# keeps underscore italics (e.g. "_AF123456_").
GENBANK_NUCLEOTIDE_RE = r'[A-Z]\d{5}|[A-Z]{2}\d{6}|[A-Z]{2}\d{8}|[A-Z]{4}\d{8,10}|[A-Z]{6}\d{9,11}'
GENBANK_PROTEIN_RE = r'[A-Z]{3}\d{5}|[A-Z]{3}\d{7}'
GENBANK_RE = re.compile(
    r'(?<![A-Za-z0-9.\-])'
    r'(' + GENBANK_NUCLEOTIDE_RE + '|' + GENBANK_PROTEIN_RE + r')'
    r'(?:\.\d+)?'
    r'(?![A-Za-z0-9\-])'
)
_GENBANK_PROTEIN_FULL_RE = re.compile(GENBANK_PROTEIN_RE)

# INSDC never assigns O, P or Q as a one-letter prefix; those are UniProt.
_UNIPROT_LIKE_RE = re.compile(r'[OPQ]\d{5}')
# WGS master records (all-zero sequence number) describe a whole assembly.
_WGS_MASTER_RE = re.compile(r'[A-Z]{4,6}\d{2}0{6,}')
# Identifiers with an accession shape that are not INSDC sequences: GEO, SRA/ENA
# runs and studies, WormBase gene and protein IDs for other nematodes (CBG12345,
# CBP12345, ...) and B. subtilis knockout strains (BKK12345). CCD is NOT here:
# CCD65109 is a real INSDC protein accession for a C. elegans gene.
NON_INSDC_PREFIXES = frozenset({
    "GSE", "GSM", "GPL", "GDS",
    "SRR", "SRX", "SRP", "SRS", "ERR", "ERX", "ERP", "ERS", "DRR", "DRX", "DRP",
    "CBG", "CBN", "CBP", "CJA", "CRE", "PPA", "BKK",
})

# Words that, near a hit, show the author meant a sequence accession.
CONTEXT_RE = re.compile(r'GenBank|EMBL|DDBJ|\bENA\b|accession|\bAcc\.|RefSeq|NCBI', re.IGNORECASE)
CONTEXT_WINDOW = 150

EUTILS_ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"


@dataclass
class AccessionHit:
    accession: str          # unversioned, e.g. "NM_013646"
    db: str                 # NCBI database to link from: "nuccore" or "protein"
    count: int = 0
    has_context: bool = False


def accession_db(accession: str) -> str:
    """Return the NCBI database ("nuccore" or "protein") an accession belongs to."""
    if "_" in accession:
        return "protein" if accession.split("_", 1)[0] in REFSEQ_PROTEIN_PREFIXES else "nuccore"
    return "protein" if _GENBANK_PROTEIN_FULL_RE.fullmatch(accession) else "nuccore"


def _is_excluded_genbank(accession: str) -> bool:
    return (
        bool(_UNIPROT_LIKE_RE.fullmatch(accession))
        or bool(_WGS_MASTER_RE.fullmatch(accession))
        or accession[:3] in NON_INSDC_PREFIXES
    )


def find_accessions(text: str) -> Dict[str, AccessionHit]:
    """Return every candidate accession in ``text``, keyed by unversioned accession.

    ``has_context`` is True when any occurrence has an accession keyword
    (GenBank, RefSeq, accession, ...) within CONTEXT_WINDOW characters.
    """
    hits: Dict[str, AccessionHit] = {}
    if not text:
        return hits
    for regex in (REFSEQ_RE, GENBANK_RE):
        for match in regex.finditer(text):
            accession = match.group(1)
            if regex is GENBANK_RE and _is_excluded_genbank(accession):
                continue
            hit = hits.get(accession)
            if hit is None:
                hit = hits[accession] = AccessionHit(accession=accession, db=accession_db(accession))
            hit.count += 1
            if not hit.has_context:
                window = text[max(0, match.start() - CONTEXT_WINDOW): match.end() + CONTEXT_WINDOW]
                hit.has_context = bool(CONTEXT_RE.search(window))
    return hits


@dataclass
class NcbiAccessionResolver:
    """Resolve accessions to NCBI Gene IDs with E-utilities elink, with an on-disk cache.

    Each accession is sent as its own ``id`` parameter, so elink returns one
    linkset per accession in request order (a comma-separated id list would merge
    all links into one set). Accessions NCBI does not know come back as an empty
    linkset and are cached as having no gene, like accessions with no gene link.
    """
    cache_path: Optional[str] = None
    batch_size: int = 50
    cache_ttl_days: int = 90
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get("NCBI_API_KEY"))
    timeout: float = 60.0
    retries: int = 3
    _cache: Dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _last_request: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, encoding="utf-8") as fh:
                    self._cache = json.load(fh)
            except (OSError, ValueError) as exc:
                logger.warning("Ignoring unreadable accession cache %s: %s", self.cache_path, exc)

    def save_cache(self):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self._cache, fh)
        os.replace(tmp_path, self.cache_path)

    def _cached(self, accession: str) -> Optional[List[int]]:
        entry = self._cache.get(accession)
        if entry is None or time.time() - entry["t"] > self.cache_ttl_days * 86400:
            return None
        return entry["genes"]

    def resolve(self, hits: Iterable[AccessionHit]) -> Dict[str, List[int]]:
        """Return {accession: [NCBI Gene IDs]} for every hit (empty list = no gene)."""
        result: Dict[str, List[int]] = {}
        to_fetch: Dict[str, List[str]] = {"nuccore": [], "protein": []}
        for hit in hits:
            cached = self._cached(hit.accession)
            if cached is not None:
                result[hit.accession] = cached
            elif hit.accession not in to_fetch[hit.db]:
                to_fetch[hit.db].append(hit.accession)
        for db, accessions in to_fetch.items():
            for start in range(0, len(accessions), self.batch_size):
                batch = accessions[start:start + self.batch_size]
                fetched = self._elink_splitting(db, batch)
                now = time.time()
                for accession in batch:
                    genes = fetched.get(accession)
                    if genes is None:
                        # Request failed; leave uncached so the next run retries.
                        result[accession] = []
                        continue
                    self._cache[accession] = {"genes": genes, "t": now}
                    result[accession] = genes
        return result

    def _throttle(self):
        min_interval = 0.11 if self.api_key else 0.35
        wait = self._last_request + min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()

    def _elink_splitting(self, db: str, accessions: List[str]) -> Dict[str, List[int]]:
        """Like _elink, but split a batch that keeps failing in half and retry each half.

        NCBI dropped the connection ("Connection broken: InvalidChunkLength") on about
        one in six 200-accession batches, all retries included, and on none of the
        50-accession batches over the same 9,346 accessions. Splitting keeps a batch that
        still fails from losing every accession in it.
        """
        fetched = self._elink(db, accessions)
        if fetched or len(accessions) == 1:
            if not fetched:
                logger.error("elink %s could not resolve %s", db, accessions[0])
            return fetched
        middle = len(accessions) // 2
        fetched = self._elink_splitting(db, accessions[:middle])
        fetched.update(self._elink_splitting(db, accessions[middle:]))
        return fetched

    def _elink(self, db: str, accessions: List[str]) -> Dict[str, List[int]]:
        """Return {accession: [gene IDs]} for one batch; {} if the request failed."""
        params = [("dbfrom", db), ("db", "gene"), ("idtype", "acc"), ("retmode", "json")]
        if self.api_key:
            params.append(("api_key", self.api_key))
        params.extend(("id", accession) for accession in accessions)
        for attempt in range(1, self.retries + 1):
            self._throttle()
            try:
                response = requests.post(EUTILS_ELINK_URL, data=params, timeout=self.timeout)
                response.raise_for_status()
                # NCBI error messages can carry raw control characters.
                payload = json.loads(re.sub(r'[\x00-\x1f]', ' ', response.text))
                return parse_elink_linksets(accessions, payload.get("linksets", []))
            except (requests.RequestException, ValueError) as exc:
                logger.warning("elink %s batch of %d failed (attempt %d/%d): %s",
                               db, len(accessions), attempt, self.retries, exc)
                time.sleep(attempt * 2)
        logger.warning("elink %s batch of %d failed after %d attempts", db, len(accessions), self.retries)
        return {}


def parse_elink_linksets(accessions: List[str], linksets: List[dict]) -> Dict[str, List[int]]:
    """Map elink linksets back to the requested accessions.

    Linksets come back in request order. A known accession echoes its versioned
    id ("NM_013646.3"), which is checked against the request; an unknown one has
    an empty ``ids`` list and an error, and is matched by position.
    """
    result: Dict[str, List[int]] = {}
    if len(linksets) != len(accessions):
        logger.warning("elink returned %d linksets for %d accessions; matching by id only",
                       len(linksets), len(accessions))
    requested = set(accessions)
    for position, linkset in enumerate(linksets):
        ids = linkset.get("ids") or []
        if ids:
            accession = str(ids[0]).split(".", 1)[0]
            if accession not in requested:
                continue
        elif len(linksets) == len(accessions):
            accession = accessions[position]
        else:
            continue
        genes: List[int] = []
        for linksetdb in linkset.get("linksetdbs") or []:
            if linksetdb.get("dbto") == "gene" and not linksetdb.get("ERROR"):
                genes.extend(int(gene_id) for gene_id in linksetdb.get("links") or [])
        result[accession] = sorted(set(genes))
    # When the linksets cannot be lined up with the request, accessions that were
    # not matched by id are left out, so the caller retries them instead of
    # caching them as having no gene.
    return result
