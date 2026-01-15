import os
import re
import unicodedata
from functools import lru_cache

# We import lazily inside functions to avoid circulars if this module is imported early.
# from utils.abc_utils import get_all_curated_entities  # (done lazily)

TOPIC_SPECIES = "ATP:0000123"


def is_species_topic(topic: str) -> bool:
    return topic == TOPIC_SPECIES


# -----------------------------------------------------------------------------
# hard-code these common-name aliases that we can't derived from curated
# species list
# -----------------------------------------------------------------------------
_COMMON_NAME_ALIASES = {
    "human": "Homo sapiens",
    "humans": "Homo sapiens",
    "mouse": "Mus musculus",
    "mice": "Mus musculus",
    "murine": "Mus musculus",
    "rat": "Rattus norvegicus",
    "rats": "Rattus norvegicus",
    "bovine": "Bos taurus",
    "cow": "Bos taurus",
    "cows": "Bos taurus",
    "calf": "Bos taurus",
    "calves": "Bos taurus",
    "zebrafish": "Danio rerio",
    "danio": "Danio rerio",
    "fruitfly": "Drosophila melanogaster",
    "fruit fly": "Drosophila melanogaster",
    "fruitflies": "Drosophila melanogaster",
    "fruit-flies": "Drosophila melanogaster",
    "fruit-fly": "Drosophila melanogaster",
    "budding yeast": "Saccharomyces cerevisiae S288C",
    "budding-yeast": "Saccharomyces cerevisiae S288C",
    "fission yeast": "Schizosaccharomyces pombe",
    "fission-yeast": "Schizosaccharomyces pombe",
}

# genus initial expansion fallback (used by expand_species_abbreviations)
_EPITHET_TO_GENUS = {
    "elegans": "Caenorhabditis",
    "melanogaster": "Drosophila",
    "rerio": "Danio",
    "musculus": "Mus",
    "norvegicus": "Rattus",
    "cerevisiae": "Saccharomyces",
    "pombe": "Schizosaccharomyces",
}


def _normalize_spaces_and_tags(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    # Bridge occasional markup-split initials like "C.</hi> elegans" -> "C. elegans"
    t = re.sub(r'([A-Z])\.\s*</[^>]+>\s*([a-z])', r'\1. \2', t)
    # Normalize spaces
    t = t.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _abbrev_variants_for_name(canonical: str):
    """
    Given 'Genus species [extra]' build common abbreviation variants:
      - 'G. species [extra]'
      - 'G.species [extra]'
      - 'G species [extra]'   (no dot; occurs in some texts)
    Only emitted if we have at least two tokens (binomial/trinomial).
    """
    toks = canonical.split()
    if len(toks) < 2:
        return []  # not a binomial/trinomial; nothing to do
    genus = toks[0]
    tail = " ".join(toks[1:])
    initial = genus[0]
    # Do not emit duplicates; return in a stable order
    variants = [
        f"{initial}. {tail}",
        f"{initial}.{tail.split()[0]}",  # 'C.elegans' (no space before epithet)
        f"{initial} {tail}",             # 'C elegans'
    ]
    # If there is more than two tokens (e.g., S288C), also add dot/no-dot with the full tail
    if len(toks) > 2:
        variants.append(f"{initial}.{tail.replace(' ', ' ')}")  # 'S.cerevisiae S288C'
    # Make unique while preserving order
    seen = set()
    out = []
    for v in variants:
        if v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def _build_alias_map_from_curated(curated_names):
    """
    Build a dict {alias -> canonical} by:
      - Adding abbreviation variants for every curated binomial/trinomial.
      - Adding the curated names themselves as identity.
    Returns (alias_map, alias_list_sorted_for_regex)
    """
    alias_map = {}

    # 1) Curated names themselves
    for name in curated_names:
        alias_map[name] = name

    # 2) Abbreviations derived from curated binomials/trinomials.
    for name in curated_names:
        for alias in _abbrev_variants_for_name(name):
            alias_map[alias] = name

    # 3) Merge in the small set of common-name aliases (hard-coded).
    for k, v in _COMMON_NAME_ALIASES.items():
        alias_map[k] = v

    # Sort aliases longest-first so the regex prefers the longest match.
    aliases_sorted = sorted(alias_map.keys(), key=len, reverse=True)
    return alias_map, aliases_sorted


@lru_cache(maxsize=1)
def _load_compiled_aliases():
    """
    Load curated species names for the selected MODs and compile one big regex.
    Control which MODs to use via env var SPECIES_ALIAS_MODS (comma-separated), default 'WB'.
    """
    mods_env = os.environ.get("SPECIES_ALIAS_MODS", "WB")
    mods = [m.strip() for m in mods_env.split(",") if m.strip()]

    curated_names = set()
    try:
        # Lazy import to avoid circulars at module import time.
        from utils.ateam_utils import get_all_curated_entities
        for mod in mods:
            names, _mapping = get_all_curated_entities(mod, "species")
            curated_names.update(names or [])
    except Exception:
        # If anything fails, fall back to just the handcrafted aliases.
        curated_names = set()

    alias_map, aliases_sorted = _build_alias_map_from_curated(curated_names)

    # Build one case-insensitive, token-bounded regex of all aliases.
    # Token boundaries: avoid inside-word matches (genes etc.).
    esc = [re.escape(a) for a in aliases_sorted]
    big = r'(?<![A-Za-z0-9])(' + '|'.join(esc) + r')(?![A-Za-z0-9])'
    pattern = re.compile(big, re.IGNORECASE)

    # For fast lookup during replacement, map lowercase(alias) -> canonical
    lower_map = {k.lower(): v for k, v in alias_map.items()}

    return pattern, lower_map


def species_alias_to_name_mapping():
    """
    Retained for backward-compat callers.
    Now returns ONLY the small hand-coded mapping (common names).
    Abbreviation variants are generated dynamically from curated names.
    """
    return dict(_COMMON_NAME_ALIASES)


def normalize_species_aliases(text: str) -> str:
    """
    This function normalizes species names or aliases in a given string (text) by
    replacing any known aliases (including dynamically generated abbreviations) with
    their canonical (curated) species names - just regex pattern matching and replacement
    """

    # Calls _normalize_spaces_and_tags to clean up extra spaces, tags, or artifacts
    # _normalize_spaces_and_tags("   C. elegans   ")  # -> "C. elegans"
    s = _normalize_spaces_and_tags(text or "")
    if not s:
        return s

    # ---------------------------------------------------------------------------------
    # Load alias pattern and lookup map
    # Loads a compiled regex pattern (pattern) to detect all known aliases
    # Loads a dictionary (lower_map) that maps lowercase aliases → canonical species name
    # Example of lower_map:
    # {
    #    "c. elegans": "Caenorhabditis elegans",
    #    "d. rerio": "Danio rerio",
    #    "h. sapiens": "Homo sapiens"
    # }
    # ---------------------------------------------------------------------------------

    pattern, lower_map = _load_compiled_aliases()

    # ---------------------------------------------------------------------------------
    # For each regex match (m), it:
    #   * Converts the match to lowercase.
    #   * Looks it up in lower_map.
    #   * If found, replaces it with the canonical name.
    #   * If not found, keeps the original text unchanged.
    # ---------------------------------------------------------------------------------
    def _repl(m):
        key = m.group(0).lower()
        return lower_map.get(key, m.group(0))

    # Uses the compiled regex pattern to search through the cleaned text s and replace every alias
    # match with its canonical version using _repl.

    return pattern.sub(_repl, s)


# ---------------------------------------------------------------------------------------
# optional fallback:
# creates a compiled regular expression that matches species abbreviations like:
# C. elegans
# D. rerio
# H. sapiens
# ---------------------------------------------------------------------------------------
_SPECIES_ABBR_RE = re.compile(r'(?<![A-Za-z0-9])([A-Z])\.\s*([a-z]{2,})(?![A-Za-z0-9])')


def expand_species_abbreviations(text: str) -> str:
    s = _normalize_spaces_and_tags(text or "")

    def repl(m):
        epithet = m.group(2).lower()
        genus = _EPITHET_TO_GENUS.get(epithet)
        return f"{genus} {epithet}" if genus else m.group(0)
    return _SPECIES_ABBR_RE.sub(repl, s)


# if we want to rebuild the alias regex mid-process (e.g. tests)
def _reset_species_alias_cache_for_tests_only():
    _load_compiled_aliases.cache_clear()
