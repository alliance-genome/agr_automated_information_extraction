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
