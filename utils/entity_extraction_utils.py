"""
entity_extraction_utils.py

Shared utilities for AGR entity extraction pipelines:
- Curated entity list caching (names + CURIEs) by (MOD, entity_type)
- Model / HF pipeline caching
- Safe batched NER
- Text prefiltering
- Title/abstract/fulltext candidate merging
- Name → CURIE mapping
- Stable cache-key computation (e.g., for API result caches)

Note: callers provide their own regex pattern and (optionally) normalizers.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Iterable, Set
from collections import Counter, defaultdict, deque
import hashlib
import json
import logging
import re
import dill
from transformers import pipeline
try:
    from transformers.modeling_utils import PreTrainedModel
except Exception:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Disk caches                                                           #
# --------------------------------------------------------------------- #
ENTITY_CACHE_DIR = Path("/data/agr_entity_extraction/cache")
ENTITY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_CACHE: Dict[Tuple[str, str], object] = {}
_PIPE_CACHE: Dict[Tuple[str, str], Optional[object]] = {}
_ENTITY_CACHE: Dict[Tuple[str, str], Tuple[List[str], Dict[str, str]]] = {}

# --------------------------------------------------------------------- #
# PATTERNS & TOPIC MAPPING                                              #
# --------------------------------------------------------------------- #
STRAIN_NAME_PATTERN = re.compile(
    r'''
    (?<![A-Za-z0-9])              # left-boundary: not preceded by a letter or digit
    (?=[A-Za-z0-9()_-]*[A-Za-z])  # must contain at least one letter
    (?=[A-Za-z0-9()_-]*\d)        # must contain at least one digit
    [A-Za-z0-9]                   # first char: letter or digit
    (?:                           # then any ofâ€¦
       [A-Za-z0-9_-]              #   letter/digit/underscore/hyphen
     | \([A-Za-z0-9]+\)           #   "(â€¦)" group of alphanumerics
    )*                            # repeated
    (?![A-Za-z0-9])               # right-boundary: next char is not a letter or digit
    ''',
    re.VERBOSE
)

GENERIC_NAME_PATTERN = re.compile(
    r'(?<![A-Za-z0-9_.\-])'          # left-delimiter: not letter/digit/._-
    r'(?=[A-Za-z0-9_.\-]*[A-Za-z])'  # must contain â‰¥1 letter
    r'(?=[A-Za-z0-9_.\-]*\d)'        # must contain â‰¥1 digit
    r'[A-Za-z0-9_.\-]{2,}'           # the token itself (no colon!)
    r'(?![A-Za-z0-9_.\-])'           # right-delimiter: not letter/digit/._-
)

# Allele-specific pattern:
# - must start with a LOWERCASE letter (C. elegans lab prefix)
# - can contain letters/digits/._-
# - must contain at least one lowercase letter and at least one digit
#   (allows mixed-case like ttTi4348, oxTi970, mgDf50, ccTi1594, etc.)
ALLELE_NAME_PATTERN = re.compile(
    r'(?<![A-Za-z0-9_.\-])'       # left-delimiter: not letter/digit/._-
    r'(?=[A-Za-z0-9_.\-]*[a-z])'  # must contain >=1 lowercase letter
    r'(?=[A-Za-z0-9_.\-]*\d)'     # must contain >=1 digit
    r'[a-z][A-Za-z0-9_.\-]{1,}'   # FIRST char lowercase; then >=1 allowed chars
    r'(?![A-Za-z0-9_.\-])'        # right-delimiter: not letter/digit/._-
)
"""
This pattern now matches
ttTi4348, ttTi5605
oxTi970, oxTi185
ccTi1594, cxTi10882, cxTi9279
mgDf50, gkDf31, mgDf90
dnSi4, juSi123, ieSi65, ltSi915
all classic alleles: e1370, e1490, n745, ok1255, tm3411, tm6119, gk218
"""

# Suspect short alleles that frequently appear as labels/panel IDs, etc.
SUSPICIOUS_SHORT_ALLELES = {f"e{i}" for i in range(1, 10)}  # e1..e9

# Any single-letter + digits pattern is suspicious (e5, s7, f1, m2, z1, etc.)
SUSPICIOUS_PREFIX_RE = re.compile(r"^[a-z]\d{1,2}$")

# Words that suggest true allele context
ALLELE_CONTEXT_WORDS = {
    "allele", "alleles",
    "mutant", "mutants",
    "mutation", "mutations",
    "variant", "variants",
    "suppressor", "suppressors",
}

# ---------------------------------------------------------------------------
# Allele false positive detection patterns
# ---------------------------------------------------------------------------

# Common background marker alleles used for transformation (allele is not being studied)
# Pattern: unc-119(ed3), unc-119 (ed3), etc.
BACKGROUND_MARKER_ALLELES = {"ed3", "ed4"}

# Genetic marker genes used in crosses (dpy=dumpy, unc=uncoordinated, etc.)
# These alleles are used as visible markers, not being studied
GENETIC_MARKER_GENES = {
    "dpy-10", "dpy-17", "dpy-20",  # dumpy markers
    "unc-4", "unc-32", "unc-52",   # uncoordinated markers
    "rol-6",                       # roller marker
    "lon-2",                       # long marker
    "lin-15",                      # multivulva marker
}
GENETIC_MARKER_ALLELES = {
    "e128", "e120",  # dpy-10(e128), unc-4(e120) from paper 8
}

# Control allele context patterns
CONTROL_CONTEXT_PATTERN = re.compile(
    r'(positive\s+control|negative\s+control|control\s+(strain|animal|worm)|'
    r'used\s+as\s+(a\s+)?control|as\s+(a\s+)?(positive|negative)?\s*control)',
    re.IGNORECASE
)

# MosSCI transposon insertion sites (reagent, not an allele being studied)
# These are specific loci used for single-copy transgene insertion
# Use anchors to ensure exact match, not prefix matching
MOSCI_INSERTION_SITE_PATTERN = re.compile(r"^ttTi\d+$", re.IGNORECASE)

# Balancer chromosome NAMES themselves (these are NOT alleles being studied)
BALANCER_CHROMOSOME_NAMES = {
    "ht2", "qc1", "nt1", "min1", "et1", "mt1", "sc1", "szt1",
    "hin1", "mnc1", "sdp2", "sdp3", "mdf1", "qdf",
}

# Common balancer allele names (used to maintain lethal mutations)
KNOWN_BALANCER_ALLELES = {
    "e937", "q782", "e1259", "q339",  # from user's examples
    "q361", "h661", "mn316", "mn398",  # other common balancers
}

# Pre-built balancer names pattern (built from BALANCER_CHROMOSOME_NAMES)
# Used in step 5b to detect alleles inside balancer constructs
# Sorted for deterministic ordering across runs
_BALANCER_NAMES_PATTERN = r'(' + '|'.join(re.escape(b) for b in sorted(BALANCER_CHROMOSOME_NAMES)) + r')'

# AID (auxin-inducible degradation) system - tir1 is a plant protein
AID_REAGENT_PATTERN = re.compile(
    r'(auxin[- ]inducible|AID\s*(system|degron)?|plant[- ]specific\s+F[- ]box|TIR1)',
    re.IGNORECASE
)

# Yeast-related context patterns
YEAST_CONTEXT_PATTERN = re.compile(
    r'(yeast|S\.\s*cerevisiae|Saccharomyces|budding\s+yeast|fission\s+yeast|'
    r'S\.\s*pombe|Schizosaccharomyces)',
    re.IGNORECASE
)
# Yeast deletion strain pattern: Δyfh1, Δade2, etc.
YEAST_DELETION_PATTERN = re.compile(r'Δ(\w+)', re.UNICODE)

# Reference-only mentions - allele mentioned only by comparison, not studied
REFERENCE_ONLY_PATTERNS = [
    re.compile(r'identical\s+to\s+(?:the\s+)?(?:\w+-?\d*\s*\()?' + r'(\w+)\)?', re.IGNORECASE),
    re.compile(r'same\s+as\s+(?:the\s+)?(?:\w+-?\d*\s*\()?' + r'(\w+)\)?', re.IGNORECASE),
    re.compile(r'equivalent\s+to\s+(?:the\s+)?(?:\w+-?\d*\s*\()?' + r'(\w+)\)?', re.IGNORECASE),
]

# Integrated transgene markers - typically used as reporters/markers, not studied
# Patterns: qIs51, wIs50, ruIs32, etc.
# Format: 1-4 letter prefix + Is + digits
# NOTE: Si (single-copy insertions like ieSi64, juSi123) and Ti (MosSCI sites like
# ttTi4348) are excluded - these can be studied alleles. Ti patterns are handled by
# step 3's context-sensitive MosSCI check instead.
TRANSGENE_MARKER_PATTERN = re.compile(r'^[a-z]{1,4}Is\d+$', re.IGNORECASE)

# Extrachromosomal array patterns (qEx*, wEx*, pzEx*, etc.) - typically constructs
EXTRACHROMOSOMAL_ARRAY_PATTERN = re.compile(r'^[a-z]{1,4}Ex\d+$', re.IGNORECASE)

# Context patterns indicating a transgene/array is used as a marker (not studied)
# If the candidate appears near these terms, it's likely a marker/reagent
MARKER_CONTEXT_KEYWORDS = (
    r'(reporter|marker|GFP|mCherry|RFP|YFP|CFP|fluorescent|'
    r'expressing|expression\s+pattern|visualize|labeled|tag(ged)?)'
)


def is_false_positive_allele(fulltext: str, candidate: str) -> tuple[bool, str]:  # noqa: C901
    """
    Check if an allele candidate is a false positive based on context analysis.

    Returns:
        (is_false_positive, reason) - True if the candidate should be filtered out,
        along with a description of why.
    """
    if not fulltext or not candidate:
        return False, ""

    cand_lower = candidate.lower()

    # 1. Check for transgene markers (qIs*, wIs*, etc.)
    # Only filter if used in marker/reporter context to avoid dropping studied transgenes
    if TRANSGENE_MARKER_PATTERN.match(candidate):
        # Check if candidate appears near marker-related context
        marker_context = re.escape(candidate) + r'.{0,50}' + MARKER_CONTEXT_KEYWORDS
        context_before = MARKER_CONTEXT_KEYWORDS + r'.{0,50}' + re.escape(candidate)
        if re.search(marker_context, fulltext, re.IGNORECASE) or \
           re.search(context_before, fulltext, re.IGNORECASE):
            return True, f"transgene marker ({candidate})"

    # 2. Check for extrachromosomal arrays (qEx*, pzEx*, etc.)
    # Only filter if used in marker/reporter context to avoid dropping studied arrays
    if EXTRACHROMOSOMAL_ARRAY_PATTERN.match(candidate):
        # Check if candidate appears near marker-related context
        marker_context = re.escape(candidate) + r'.{0,50}' + MARKER_CONTEXT_KEYWORDS
        context_before = MARKER_CONTEXT_KEYWORDS + r'.{0,50}' + re.escape(candidate)
        if re.search(marker_context, fulltext, re.IGNORECASE) or \
           re.search(context_before, fulltext, re.IGNORECASE):
            return True, f"extrachromosomal array ({candidate})"

    # 3. Check for MosSCI transposon insertion sites (ttTi4348, ttTi5605, etc.)
    if MOSCI_INSERTION_SITE_PATTERN.match(candidate):
        # Verify THIS SPECIFIC candidate is used as an insertion site, not studied as an allele
        # Must match the exact candidate, not just any ttTi pattern in the text
        site_pattern = re.escape(candidate) + r'\s*(transposon|insertion\s*site|locus|site\s+on\s+chromosome)'
        if re.search(site_pattern, fulltext, re.IGNORECASE):
            return True, f"MosSCI transposon insertion site ({candidate})"

    # 4a. Check for background marker alleles (ed3, ed4 in unc-119 context)
    if cand_lower in BACKGROUND_MARKER_ALLELES:
        # Look for pattern like "unc-119(ed3)" or "unc-119 (ed3)"
        marker_pattern = r'unc-?119\s*\(\s*' + re.escape(cand_lower) + r'\s*\)'
        if re.search(marker_pattern, fulltext, re.IGNORECASE):
            # If it appears at least once in unc-119 context, it's likely a marker
            # (relaxed from 70% threshold - these alleles are almost always markers)
            return True, f"background marker ({candidate} in unc-119 context)"

    # 4b. Check for genetic marker alleles (e120, e128 in dpy-10, unc-4 context)
    if cand_lower in GENETIC_MARKER_ALLELES:
        # Look for pattern like "dpy-10(e128)" or "unc-4(e120)"
        for marker_gene in GENETIC_MARKER_GENES:
            marker_pattern = re.escape(marker_gene) + r'\s*\(\s*' + re.escape(cand_lower) + r'\s*\)'
            if re.search(marker_pattern, fulltext, re.IGNORECASE):
                return True, f"genetic marker ({candidate} in {marker_gene} context)"

    # 4c. Check for control alleles (explicitly mentioned as control)
    # Look for the allele mentioned near "control" keywords
    # Use word boundaries to avoid matching substrings (e.g., 'al2' inside 'val2')
    # Note: No DOTALL flag - window should stay within a single line to avoid
    # false positives from unrelated text across line breaks
    # Use frequency heuristic: only filter if control context is the dominant usage
    control_window_pattern = r'.{0,50}\b' + re.escape(cand_lower) + r'\b.{0,50}'
    control_mentions = 0
    for match in re.finditer(control_window_pattern, fulltext, re.IGNORECASE):
        window = match.group(0)
        if CONTROL_CONTEXT_PATTERN.search(window):
            control_mentions += 1
    if control_mentions > 0:
        # Count total mentions of this allele
        allele_pattern = r'\b' + re.escape(cand_lower) + r'\b'
        total_mentions = len(re.findall(allele_pattern, fulltext, re.IGNORECASE))
        # Only filter if ALL mentions are in control context, or if control context
        # is the dominant usage. Require at least 1 non-control mention to keep.
        non_control_mentions = total_mentions - control_mentions
        if non_control_mentions == 0 or control_mentions > total_mentions // 2:
            return True, f"control allele ({candidate})"

    # 5a. Check if the candidate IS a balancer chromosome name itself (qC1, hT2, nT1, etc.)
    if cand_lower in BALANCER_CHROMOSOME_NAMES:
        # Verify it's used as a balancer (appears with [...] or /+ or as a strain name)
        balancer_usage_pattern = r'\b' + re.escape(cand_lower) + r'(/\+)?\s*(\[|with|strain|balancer)'
        if re.search(balancer_usage_pattern, fulltext, re.IGNORECASE):
            return True, f"balancer chromosome name ({candidate})"

    # 5b. Check for balancer chromosome alleles (alleles INSIDE balancer constructs)
    if cand_lower in KNOWN_BALANCER_ALLELES:
        # Look for balancer patterns and check if allele appears near them
        # Pattern matches: hT2[...e937...], qC1[...e1259...], hT2/+ [...e937...]
        # Use a window-based approach to handle nested brackets
        # Look for balancer name followed by the allele within ~150 chars
        # Exclude semicolons, newlines, and periods to avoid spanning clause/sentence boundaries
        # Use word boundaries on both sides to avoid substring matches (e.g., ze937)
        balancer_window_pattern = _BALANCER_NAMES_PATTERN + r'[^;\n.]{0,150}\b' + re.escape(cand_lower) + r'\b'
        if re.search(balancer_window_pattern, fulltext, re.IGNORECASE):
            return True, f"balancer allele ({candidate})"

    # 6. Check for AID/TIR1 system (plant protein, not C. elegans allele)
    if cand_lower == "tir1":
        if AID_REAGENT_PATTERN.search(fulltext):
            return True, f"AID system reagent ({candidate} is plant TIR1)"

    # 7. Check for yeast strains/genes (not C. elegans)
    # Look for yeast deletion pattern Δyfh1 - check ALL matches, not just the first
    for yeast_m in YEAST_DELETION_PATTERN.finditer(fulltext):
        if yeast_m.group(1).lower() == cand_lower:
            if YEAST_CONTEXT_PATTERN.search(fulltext):
                return True, f"yeast gene/strain ({candidate})"
            break

    # Also check if the candidate appears specifically in yeast context
    # Look for patterns like "yfh1 yeast" or "S. cerevisiae yfh1"
    # NOTE: Do NOT include generic "strain" here - it's used for C. elegans strains too
    yeast_allele_pattern = (
        r'(' + re.escape(cand_lower) + r'\s+(yeast|S\.\s*cerevisiae)|'
        r'(yeast|S\.\s*cerevisiae)\s+' + re.escape(cand_lower) + r')'
    )
    if re.search(yeast_allele_pattern, fulltext, re.IGNORECASE):
        return True, f"yeast gene/strain ({candidate})"

    # 8. Check for reference-only mentions ("identical to X allele")
    # Check ALL matches per pattern, not just the first
    # Note: Each pattern in REFERENCE_ONLY_PATTERNS is checked independently.
    # A candidate could be filtered by "same as" even if "identical to" didn't match.
    for pattern in REFERENCE_ONLY_PATTERNS:
        for match in pattern.finditer(fulltext):
            referenced_allele = match.group(1).lower()
            if referenced_allele == cand_lower:
                # Verify this is the primary/only context for this allele
                # Count total mentions of this allele in the text
                allele_pattern = r'\b' + re.escape(cand_lower) + r'\b'
                allele_count = len(re.findall(allele_pattern, fulltext, re.IGNORECASE))
                if allele_count <= 3:  # Very few mentions suggests reference-only
                    return True, f"reference-only mention ({candidate})"
                # Found candidate in this pattern but count is high; skip remaining
                # matches for this pattern but continue checking other patterns
                break

    return False, ""


def filter_false_positive_alleles(
    entities: list[str],
    fulltext: str,
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Filter out false positive alleles from a list of candidate entities.

    Args:
        entities: List of candidate allele names
        fulltext: Full text of the paper for context analysis

    Returns:
        (filtered_entities, rejected_with_reasons) - List of entities that passed
        filtering, and list of (entity, reason) tuples for rejected ones.
    """
    # Warn if no fulltext available - all entities will pass through unfiltered
    if not fulltext:
        logger.warning("ALLELE-FP-FILTER: no fulltext available, skipping false positive filtering")
        return entities, []

    filtered = []
    rejected = []

    for ent in entities:
        is_fp, reason = is_false_positive_allele(fulltext, ent)
        if is_fp:
            rejected.append((ent, reason))
            logger.info("ALLELE-FP-FILTER: rejecting '%s' - %s", ent, reason)
        else:
            filtered.append(ent)

    return filtered, rejected


# --------------------------------------------------------------------- #
# Threshold-tuning gold sets (restored)                                 #
# --------------------------------------------------------------------- #
STRAIN_TARGET_ENTITIES = {
    "AGRKB:101000000641073": ["N2", "OP50", "TJ375"],
    "AGRKB:101000000641132": ["EG4322", "GE24"],
    "AGRKB:101000000641112": ["N2", "EG6699", "JT734"],
    "AGRKB:101000000640598": [
        "XZ1515", "XZ1514", "QX1794", "PB306", "ECA369", "CX11271", "JT73", "JT513", "XZ1513",
        "JJ1271", "ECA36", "SP346", "RB2488", "ECA372", "NIC268", "RB1658", "NH646", "LKC34",
        "CB185", "JU1200", "RB1977", "ECA189", "JU258", "XZ1516", "JU367", "GH383", "CX11314",
        "QG556", "ECA191", "NIC256", "RT362", "WN2001", "MY10", "JU775", "BA819", "CB4932",
        "PB303", "JK4545", "OP50", "NIC251", "JU1242", "QG2075", "CB30", "GL302", "QX1791",
        "ECA396", "JT11398", "JU830", "JU363", "QX1793", "EG4725", "NIC199", "CB4856",
        "ECA363", "N2"
    ],
    "AGRKB:101000000641062": ["PD1074", "HT115"],
    "AGRKB:101000000641018": ["VC2428", "N2", "OP50", "VC1743"],
    "AGRKB:101000000640727": [
        "VC1263", "CB3203", "CB3257", "MT5013", "SP1713", "VC610", "CB3261", "MT5006",
        "RB983", "MT4433", "MT8886", "KJ462", "MT9958", "PR678", "CB936", "N2", "CU1715",
        "NG144", "RB1100", "NF87", "CU2945", "PR811", "PR691", "MT11068", "MT4434", "PR767"
    ],
    "AGRKB:101000000640813": ["N2", "CB4856", "JU1580"],
    "AGRKB:101000000639765": [
        "N2", "DA1814", "AQ866", "LX702", "LX703", "CX13079", "OH313", "VC125", "VC670",
        "RB785", "RB1680"
    ],
    "AGRKB:101000000640768": ["KG1180", "RB830", "TR2171", "ZX460", "OP50-1"]
}

ALLELE_TARGET_ENTITIES = {
    "AGRKB:101000001029627": ['km25', 'mu86', 'ok434', 'ok524', 'zu135'],
    "AGRKB:101000001033554": [
        'e1393', 'e1398', 'e502', 'ev554', 'jj278', 'm40', 'm62',
        'm77', 'ok3336', 'ok3416', 'tm2091', 'wk30', 'wk7', 'wk70'
    ],
    "AGRKB:101000001052808": [
        'ad465', 'km21', 'km25', 'mu86', 'nk3', 'nr2041', 'ok1255', 'ok186',
        'ok2065', 'ok3730', 'ok386', 'ok524', 'p675', 'qm150', 'qm30',
        'rh50', 'tm1978'
    ],
    "AGRKB:101000000638747": ['ok161'],
    "AGRKB:101000001185203": ['re257'],
    "AGRKB:101000000623393": [
        'e102', 'e234', 'e81', 'js115', 'md1088', 'md247', 'md299', 's69'
    ],
    "AGRKB:101000000623933": ['km41', 'tm1779', 'tm1898'],
    "AGRKB:101000000387847": ['gk203', 'hx546', 'ok976'],
    "AGRKB:101000000624886": ['gk349', 'ok705', 'tm843', 'tm1920', 'tm2183', 'zc32'],
    "AGRKB:101000000625317": [
        'ev546', 'n324', 'ok255', 'ok2226', 'tr64', 'tr96', 'tr101', 'tr103',
        'tr113', 'tr129', 'tr150', 'tr162', 'tr163', 'tr171', 'tr187', 'tr189'
    ],
    "AGRKB:101000000625751": [
        'bz189', 'e1735', 'e1752', 'k149', 'mu38', 'n1043', 'n1812', 'n1813', 'n1963',
        'n1993', 'n1994', 'n1995', 'n1996', 'n2433', 'n2433', 'n2438', 'n2690',
        'n3246', 'n4039', 'n432', 'n717', 'ok300', 'op149', 'op234', 'op360', 'oz167',
        't1875', 'tm1826', 'tm1949', 'tm3701'
    ],
    "AGRKB:101000000627138": ['km19', 'km21', 'ok2531', 'ok364', 'tm737'],
    "AGRKB:101000000627490": ['tm2530', 'tm2725'],
    "AGRKB:101000000639903": ['mu86', 'ra102'],
    "AGRKB:101000000960091": ['mu86', 'tm1783'],
    "AGRKB:101000001048559": ['ok2409', 'pk610', 'tm10970']
}

GENE_TARGET_ENTITIES = {
    "AGRKB:101000000634691": ["lov-3"],
    "AGRKB:101000000635145": ["his-58"],
    "AGRKB:101000000635933": ["spe-4", "spe-6", "spe-8", "swm-1", "zipt-7.1", "zipt-7.2"],
    "AGRKB:101000000635973": ["dot-1.1", "hbl-1", "let-7", "lin-29A", "lin-41", "mir-48", "mir-84", "mir-241"],
    "AGRKB:101000000636039": ["fog-3"],
    "AGRKB:101000000636419": [],
    "AGRKB:101000000637655": ["ddi-1", "hsf-1", "pas-1", "pbs-2", "pbs-3", "pbs-4", "pbs-5", "png-1", "rpt-3", "rpt-6",
                              "rpn-1", "rpn-5", "rpn-8", "rpn-9", "rpn-10", "rpn-11", "sel-1", "skn-1", "unc-54"],
    "AGRKB:101000000637658": ["dbt-1"],
    "AGRKB:101000000637693": ["C17D12.7", "plk-1", "spd-2", "spd-5"],
    "AGRKB:101000000637713": ["cha-1", "daf-2", "daf-16"],
    "AGRKB:101000000637764": ["F28C6.8", "Y71F9B.2", "acl-3", "crls-1", "drp-1", "fzo-1", "pgs-1"],
    "AGRKB:101000000637890": ["atg-13", "atg-18", "atg-2", "atg-3", "atg-4.1", "atg-4.2", "atg-7", "atg-9", "atg-18",
                              "asp-10", "bec-1", "ced-13", "cep-1", "egl-1", "epg-2", "epg-5", "epg-8", "epg-9",
                              "epg-11", "glh-1", "lgg-1", "pgl-1", "pgl-3", "rrf-1", "sepa-1", "vet-2", "vet-6",
                              "vha-5", "vha-13", "ZK1053.3"],
    "AGRKB:101000000637968": ["avr-15", "bet-2", "csr-1", "cye-1", "daf-12", "drp-1", "ego-1", "hrde-1", "lin-13",
                              "ncl-1", "rrf-1", "snu-23", "unc-31"],
    "AGRKB:101000000638021": [],
    "AGRKB:101000000638052": ["cept-1", "cept-2", "daf-22", "drp-1", "fat-1", "fat-2", "fat-3", "fat-4", "fat-6",
                              "fat-7", "fzo-1", "pcyt-1", "seip-1"],
}


def _entity_cache_path(mod_abbr: str, entity_type: str) -> Path:
    return ENTITY_CACHE_DIR / f"{mod_abbr}_{entity_type}.json"


def get_all_curated_entities_cached(
    mod_abbreviation: str,
    entity_type_str: str,
    loader_fn: Callable[[str, str], Tuple[List[str], Dict[str, str]]],
) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns (names, mapping) for a (MOD, entity_type) from disk or via loader_fn.
    """
    key = (mod_abbreviation, entity_type_str)
    if key in _ENTITY_CACHE:
        return _ENTITY_CACHE[key]

    cache_file = _entity_cache_path(mod_abbreviation, entity_type_str)
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        names = data["names"]
        mapping = data["mapping"]
        _ENTITY_CACHE[key] = (names, mapping)
        return names, mapping

    names, mapping = loader_fn(mod_abbreviation, entity_type_str)
    with cache_file.open("w", encoding="utf-8") as fh:
        json.dump({"names": names, "mapping": mapping}, fh)
    _ENTITY_CACHE[key] = (names, mapping)
    return names, mapping


def prime_model_entities(
    model: object,
    mod_abbr: str,
    entity_type: str,
    loader_fn: Callable[[str, str], Tuple[List[str], Dict[str, str]]],
) -> None:
    """
    Populate model.{entities_to_extract, name_to_curie_mapping, upper_to_original_mapping}.
    """
    names, mapping = get_all_curated_entities_cached(mod_abbr, entity_type, loader_fn)
    model.entities_to_extract = names
    model.name_to_curie_mapping = mapping
    model.upper_to_original_mapping = {n.upper(): n for n in names}
    model.alliance_entities_loaded = True


def get_model(mod_abbr: str, topic: str, path: str):
    k = (mod_abbr, topic)
    if k not in _MODEL_CACHE:
        with open(path, "rb") as fh:
            _MODEL_CACHE[k] = dill.load(fh)
    model = _MODEL_CACHE[k]
    model.topic = topic
    return model


def get_pipe(mod_abbr: str, topic: str, model: object):
    """
    Return a HF token-classification pipeline if the model is a TokenClassification model.
    Otherwise return None (regex/dictionary only).
    Note that it is a HF model..
    """
    k = (mod_abbr, topic)
    if k in _PIPE_CACHE:
        return _PIPE_CACHE[k]

    if isinstance(model, PreTrainedModel) and model.__class__.__name__.endswith("ForTokenClassification"):
        _PIPE_CACHE[k] = pipeline(
            "ner",
            model=model,
            tokenizer=model.tokenizer,
            aggregation_strategy="simple",
        )
    else:
        _PIPE_CACHE[k] = None
    return _PIPE_CACHE[k]


def run_ner_batched(pipe, texts: List[str], batch_size: int):
    """
    run_ner_batched() runs Named Entity Recognition (NER) on a list of
    texts using a Hugging Face pipeline. It tries to batch process the texts
    for speed — but safely falls back to per-text mode if the tokenizer
    cannot handle padding (which some models don’t support).
    """
    if pipe is None:
        # If no NER pipeline was passed (e.g., initialization failed),
        # return an empty list of entities for each input text.
        return [[] for _ in texts]

    # Get the tokenizer object used by the pipeline
    tok = pipe.tokenizer

    # If the tokenizer doesn’t define a padding token (needed for batching),
    # then we check alternatives.
    if getattr(tok, "pad_token_id", None) is None:
        # If the tokenizer has an EOS (end-of-sequence) token,
        # we reuse it as the padding token.
        # the EOS token is usually safe for padding in encoder-only models.
        if getattr(tok, "eos_token_id", None) is not None:
            tok.pad_token_id = tok.eos_token_id
            tok.pad_token = getattr(tok, "eos_token", None) or tok.convert_ids_to_tokens(tok.eos_token_id)
        else:
            # If there’s no EOS token either, batching isn’t possible at all
            logger.debug("No pad_token_id; running NER per-text.")
            return [pipe(t) for t in texts]
    return pipe(texts, batch_size=batch_size)


def build_parent_reverse_index(parent_children: Dict[str, Iterable[str]]) -> Dict[str, Set[str]]:
    """
    Build a reverse index: child -> {parents}. Multiple parents are allowed.
    """
    rev: Dict[str, Set[str]] = defaultdict(set)
    for p, kids in (parent_children or {}).items():
        for k in (kids or []):
            rev[k].add(p)
    return rev


def prune_to_most_specific(
    curies: List[str],
    rev_index: Dict[str, Set[str]],
) -> Tuple[List[str], Set[str]]:
    """
    Keep only the most specific taxa (drop any CURIE that is an ancestor of another).
    Returns:
      kept (list in original order), dropped_ancestors (set)
    """
    if not curies:
        return [], set()

    have = set(curies)
    ancestors: Set[str] = set()

    # climb parents transitively for each detected curie
    for c in have:
        q = deque(rev_index.get(c, ()))
        while q:
            a = q.popleft()  # removes from left end efficiently
            if a in ancestors:
                continue
            ancestors.add(a)
            # continue walking up from this ancestor
            q.extend(rev_index.get(a, ()))

    kept: List[str] = []
    seen: Set[str] = set()
    for c in curies:  # preserve original order
        if c not in ancestors and c not in seen:
            seen.add(c)
            kept.append(c)

    dropped = (have & ancestors)
    return kept, dropped


def curies_to_display_names(name_to_curie: Dict[str, str], curies: List[str]) -> List[str]:
    # Retrieve cached inverse map. Here, _inv is used as a cache to store the
    # inverse dictionary {curie → name} so it doesn’t have to rebuild it every call.
    inv = getattr(curies_to_display_names, "_inv", None)
    # If the dict identity changed, rebuild and cache it on the function
    # _src_id is the ID (memory address) of the source dictionary name_to_curie
    if inv is None or getattr(curies_to_display_names, "_src_id", None) is not id(name_to_curie):
        inv = {v: k for k, v in (name_to_curie or {}).items()}
        curies_to_display_names._inv = inv
        curies_to_display_names._src_id = id(name_to_curie)
    return [inv.get(c, c) for c in curies]


# ------------------------ Text & matching helpers --------------------- #
def extract_entities_from_title_abstract(
    model: object,
    title: str,
    abstract: str,
    pattern: re.Pattern,
) -> Tuple[List[str], List[str]]:
    """
    looks for known entity names (from a curated list) inside a paper’s
    title and abstract using both token matching and regex pattern matching.
    - Tokenizer-based matching — checks if any tokens in the text match a
      curated entity list.
    - Regex-based matching — catches patterns (like gene names or allele
      symbols) missed by tokenization.
    """
    gold = set(getattr(model, "entities_to_extract", []) or [])
    gold_up = {g.upper() for g in gold}

    tok_title = model.tokenizer.tokenize(title or "")
    ents_title = (set(tok_title) & gold) | ({t.upper() for t in tok_title if t is not None} & gold_up)
    for m in pattern.findall(title or ""):
        mu = m.upper()
        if mu in gold_up:
            ents_title.add(mu)

    tok_abs = model.tokenizer.tokenize(abstract or "")
    ents_abs = (set(tok_abs) & gold) | ({t.upper() for t in tok_abs if t is not None} & gold_up)
    for m in pattern.findall(abstract or ""):
        mu = m.upper()
        if mu in gold_up:
            ents_abs.add(mu)

    return list(ents_title), list(ents_abs)


def prefilter_text(
    fulltext: str,
    model: object,
    pattern: re.Pattern,
    *,
    is_species: bool = False,
    normalize_aliases: Optional[Callable[[str], str]] = None,
    expand_abbrevs: Optional[Callable[[str], str]] = None,
    use_gold_substring: bool = True,

) -> str:
    """
    this function keeps only the sentences or paragraphs likely to contain
    entities (like genes, strains, or species) before running expensive NER
    models.
      - match provided pattern OR
      - contain known entity substrings (case-insensitive check)
    This is used before entity extraction to:
      - Speed up NER (less text to process)
      - Reduce noise (irrelevant sentences are dropped)
    """
    if not fulltext:
        return ""

    """
    When dealing with species recognition, it may normalize:
    Aliases (e.g. "S. cerevisiae" -> "Saccharomyces cerevisiae")
    Abbreviations (e.g. "E. coli" → "Escherichia coli")
    """
    if is_species and normalize_aliases and expand_abbrevs:
        fulltext = normalize_aliases(fulltext)
        fulltext = expand_abbrevs(fulltext)

    """
    Splits text into chunks whenever it sees punctuation like
    ., ?, or ! followed by whitespace - split text using sentence boundaries.
    """
    pieces = re.split(r'(?<=[\.\?\!])\s+', fulltext)
    kept = []
    # Only compute gold_upper if we actually need it
    gold_up = {e.upper() for e in getattr(model, "entities_to_extract", []) or []} if use_gold_substring else None

    for p in pieces:
        if pattern.search(p):
            kept.append(p)
            continue
        if use_gold_substring:
            up = p.upper()
            if any(g in up for g in gold_up):
                kept.append(p)
    return " ".join(kept) if kept else fulltext


def count_curated_mentions(text: str, model: object, *, gold_up=None, allowed=None) -> "Counter[str]":
    """
    Count occurrences of curated entities in text using model.tokenizer.pattern.
    If 'allowed' is provided, only count tokens in that (usually small) set.
    Otherwise count any token in 'gold_up' (full curated set).
    """
    if not text:
        return Counter()
    pat = getattr(model.tokenizer, "pattern", None)
    if pat is None:
        return Counter()

    if gold_up is None:
        gold_up = {e.upper() for e in getattr(model, "entities_to_extract", []) or []}

    counts = Counter()
    use_allowed = allowed is not None
    if use_allowed:
        # Make membership checks fast
        allowed = set(allowed)

    for m in pat.finditer(text):
        tok = m.group(1) if m.lastindex else m.group(0)
        key = tok.upper()
        if use_allowed:
            if key in allowed:
                counts[key] += 1
        else:
            if key in gold_up:
                counts[key] += 1
    return counts


def names_to_curies(model: object, names: List[str]) -> List[str]:
    """
    Map extracted display names -> CURIEs using the model cache.
    """
    mapping = getattr(model, "name_to_curie_mapping", {}) or {}
    u2o = getattr(model, "upper_to_original_mapping", {}) or {}
    out: List[str] = []
    for n in names:
        key = n if n in mapping else u2o.get(n.upper(), n)
        curie = mapping.get(key)
        if curie:
            out.append(curie)
    return out


def resolve_entity_curie(model: object, ent: str, *, strict: bool = True) -> Optional[str]:
    """
    Map one DISPLAY NAME (possibly wrong-cased) -> CURIE.
    - strict=True: raise KeyError if no mapping
    - strict=False: return None and log a warning.
    """
    mapping = getattr(model, "name_to_curie_mapping", {}) or {}
    u2o = getattr(model, "upper_to_original_mapping", {}) or {}
    key = ent if ent in mapping else u2o.get(ent.upper())
    curie = mapping.get(key) if key else None
    if curie:
        return curie
    if strict:
        raise KeyError(f"Unknown entity with no CURIE mapping: '{ent}'")
    logger.warning("No CURIE mapping for entity '%s'", ent)
    return None


def build_entities_from_results(
    results: List[dict],
    title: str,
    abstract: str,
    fulltext: str,
    model: object,
    pattern: re.Pattern,
    *,
    is_species: bool = False,
    normalize_aliases: Optional[Callable[[str], str]] = None,
    expand_abbrevs: Optional[Callable[[str], str]] = None,
    use_fulltext_tokenizer: bool = True,
    use_count_gate: bool = True,
    use_gold_substring: bool = False,
) -> List[str]:
    """
    Merge candidates from multiple detectors (NER, regex, tokenizer, substring)
    and then optionally gates them by how many times they appear. Finally, it
    returns display names (original casing) for the curated entities
    it believes are present:
    Apply min_hits using model.min_matches (default 1) when use_count_gate=True.
    Return DISPLAY NAMES (original casing from curated list).
    """
    # 1) Normalize (species only)
    if is_species and normalize_aliases and expand_abbrevs:
        title_norm = expand_abbrevs(normalize_aliases(title or ""))
        abstract_norm = expand_abbrevs(normalize_aliases(abstract or ""))
        fulltext_norm = expand_abbrevs(normalize_aliases(fulltext or ""))
    else:
        title_norm, abstract_norm, fulltext_norm = (title or ""), (abstract or ""), (fulltext or "")

    # 2) Prep curated lookups
    gold = getattr(model, "entities_to_extract", []) or []
    gold_up = {e.upper() for e in gold}

    # 3) HF NER (accepts either 'entity_group' or 'entity')
    ents_full_hf = {
        (r.get("word") or "").upper()
        for r in (results or [])
        if (r.get("entity_group") == "ENTITY" or r.get("entity") == "ENTITY")
        and r.get("word")
        and r["word"].upper() in gold_up
    }

    # 4) Title/Abstract (tokenizer ∩ curated + regex ∩ curated)
    t_ents, a_ents = extract_entities_from_title_abstract(model, title_norm, abstract_norm, pattern)
    ents_title = {e.upper() for e in t_ents}
    ents_abs = {e.upper() for e in a_ents}

    # 5) Regex over normalized fulltext
    regex_hits = {
        m.upper()
        for m in pattern.findall(fulltext_norm or "")
        if m and m.upper() in gold_up
    }

    # 6) Optional tokenizer over normalized fulltext
    if use_fulltext_tokenizer:
        tok_full = model.tokenizer.tokenize(fulltext_norm or "")
        ents_full_tok = {t.upper() for t in tok_full if t and t.upper() in gold_up}
    else:
        ents_full_tok = set()

    # 7) Optional cheap substring fallback on short title+abstract (bounded for safety)
    #    This helps catch odd punctuation/hyphenation that slip past the regex/tokenizer.
    substring_hits = set()
    if use_gold_substring:
        small_text = (f"{title_norm} {abstract_norm}").upper()
        # only attempt if texts are short-ish and curated set isn't massive
        if small_text and len(small_text) <= 4000 and len(gold_up) <= 5000:
            # Filter out very short candidates to reduce noise/cost
            for g in gold_up:
                if len(g) >= 3 and g in small_text:
                    substring_hits.add(g)

    # 8) Union of all uppercase candidates
    all_up = ents_full_hf | ents_title | ents_abs | regex_hits | ents_full_tok | substring_hits
    if not all_up:
        return []

    # 9) Optional count gate (only compute counts when it can change the outcome)
    passed_up: set[str]
    if use_count_gate:
        min_hits = int(getattr(model, "min_matches", 1))
        if min_hits > 1:
            counts = count_curated_mentions(title_norm, model)
            counts += count_curated_mentions(abstract_norm, model)
            counts += count_curated_mentions(fulltext_norm, model)
            passed_up = {u for u in all_up if counts.get(u, 0) >= min_hits}
        else:
            passed_up = all_up
    else:
        passed_up = all_up

    if not passed_up:
        return []

    # 10) Map back to original casing using the precomputed mapping
    u2o = getattr(model, "upper_to_original_mapping", {}) or {}
    out = [u2o.get(u, u) for u in sorted(passed_up)]
    return out


def compute_cache_key(items: Iterable[str]) -> str:
    """
    Stable short SHA1-based key for caching across runs.
    Accepts any iterable of strings (e.g., curated CURIEs). De-dupes & sorts first.
    """
    dedup_sorted = sorted(set(items))
    raw = json.dumps(dedup_sorted, sort_keys=True).encode("utf-8")

    """
    Uses SHA-1 to get a fixed-length hexadecimal digest.
    Then truncates it to the first 16 characters
    >>> compute_cache_key(["BRCA1", "TP53"])
    '9f6c1e3f8577c1d2'
    """
    return hashlib.sha1(raw).hexdigest()[:16]


def has_allele_like_context(fulltext: str, candidate: str) -> bool:
    """
    True if the short allele candidate appears in a context that looks like
    a real allele, e.g. gene-1(e5), "the e5 mutant", etc.

    For *very short* suspicious alleles (one-letter + digits, e.g. "b2", "e5"),
    we are stricter: we only accept them if they appear in a gene(e5)-style pattern.
    """
    if not fulltext or not candidate:
        return False

    text = fulltext.lower()
    cand = candidate.lower()

    # Is this a "suspicious" ultra-short allele like e5, b2, s7, etc.?
    is_suspicious_short = bool(SUSPICIOUS_PREFIX_RE.match(cand))

    # gene-1(e5) / foo-1(e5)-style: gene-like token followed by (cand)
    gene_like_pattern = re.compile(
        r"[A-Za-z][A-Za-z0-9_.-]{1,15}\(" + re.escape(cand) + r"\)"
    )

    if is_suspicious_short:
        # For ultra-short patterns (e5, b2, s7, ...), require a strong
        # gene(e5)-style context. This avoids picking up figure labels like "B2".
        if gene_like_pattern.search(text):
            return True
        return False

    # For normal-length alleles (tm1783, ok1255, n324, etc.), keep your
    # original behavior: gene-like pattern OR nearby context words.
    if gene_like_pattern.search(text):
        return True

    # Simple window around the first occurrence: look for context words nearby
    idx = text.find(cand)
    if idx == -1:
        return False

    window_start = max(0, idx - 60)
    window_end = min(len(text), idx + len(cand) + 60)
    window = text[window_start:window_end]

    if any(word in window for word in ALLELE_CONTEXT_WORDS):
        return True

    return False


def rescue_short_alleles_from_fulltext(
    fulltext: str,
    model,
    already_found: set[str],
) -> set[str]:
    """
    Rescue allele names that appear in the full text but were missed by NER.

    Behavior:
    - Scan the ORIGINAL-CASE fulltext with ALLELE_NAME_PATTERN
      (which itself only matches lowercase alleles like e1370, ok1255, tm1949, etc.).
    - Ignore anything that:
        * is already in `already_found` (case-insensitive), or
        * starts with a digit (e.g. "1a", "5b-5d", "100x", "1-octanol").
    - For one-letter+digits patterns (e5, e7, b2, ...), require allele-like context
      via has_allele_like_context().
    - Returns canonical lowercase allele names.
    """
    rescued: set[str] = set()
    if not fulltext:
        return rescued

    # Track what we've already found (case-insensitive)
    already_found_lower = {e.lower() for e in (already_found or set())}

    # IMPORTANT: we now search the original-case text, not fulltext.lower().
    # ALLELE_NAME_PATTERN is lowercase-only, so it will only match true
    # lowercase allele-like tokens (e1370, ok1255, tm1949, n324, etc.),
    # and will NOT match uppercase tokens like Mos1, CD31, B2.
    for m in ALLELE_NAME_PATTERN.finditer(fulltext):
        raw = m.group(0) or ""
        if not raw:
            continue

        # Remove trailing '.' from sentence boundaries (e.g. "e1370.")
        name = raw.rstrip(".")
        if len(name) < 2:
            continue

        name_lc = name.lower()

        # Skip if we've already got this allele (case-insensitive)
        if name_lc in already_found_lower:
            continue

        # Drop anything that starts with a digit: panel labels, 100x, 1a, 2b-2c, etc.
        if name_lc[0].isdigit():
            continue

        # Suspicious if it looks like one-letter+digits: e5, e7, b2, s7, ...
        is_suspicious = (
            name_lc in SUSPICIOUS_SHORT_ALLELES
            or SUSPICIOUS_PREFIX_RE.match(name_lc) is not None
        )

        if is_suspicious:
            # Require allele-like context to keep it (gene-1(e5), "e5 mutant", etc.)
            if not has_allele_like_context(fulltext, name_lc):
                continue

        rescued.add(name_lc)

    return rescued
