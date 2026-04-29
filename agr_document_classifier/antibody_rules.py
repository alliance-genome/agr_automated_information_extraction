"""Pure-function antibody string-matching rule engine.

Ported from the Caltech WB antibody pipeline (entity-extraction-antibody/main.py).
The rules are MOD-agnostic by data; the caller supplies the curated gene list.
"""

import itertools
import re
from typing import Iterable, NamedTuple


EXCLUDE_GENES = ['PDI']
ADDITIONAL_ANTI_KEYWORDS = ['MSP']
ADDITIONAL_KEYWORDS = ['MH46', 'SP56', 'a-SP56']

COMBINATION_1 = [
    "preparation", "prepared", "prepare", "production", "purification",
    "generation", "generate", "generated", "produce", "produced",
    "purify", "purified", "raised",
]
COMBINATION_2 = ["antiserum", "antibody", "antibodies", "antisera"]


class AntibodyRules(NamedTuple):
    anti_gene_regex: re.Pattern
    additional_keywords_regex: re.Pattern
    combinations_regex: list  # list[tuple[(str, str), re.Pattern, re.Pattern]]


def build_regexes(curated_gene_names: Iterable[str]) -> AntibodyRules:
    """Compile the three rule sets once. Reuse across many sentences/papers.

    `curated_gene_names` is the WB gene list (any case — we lowercase here).
    EXCLUDE_GENES are removed; ADDITIONAL_ANTI_KEYWORDS are appended.
    """
    excluded_lower = {g.lower() for g in EXCLUDE_GENES}
    gene_names = {g.lower() for g in curated_gene_names if g.lower() not in excluded_lower}
    gene_names.update(g.lower() for g in ADDITIONAL_ANTI_KEYWORDS)
    escaped_genes = sorted(re.escape(g) for g in gene_names)

    if escaped_genes:
        anti_gene_pattern = (
            r"(?i)[\s\(\[\{\.,;:\'\"\<](anti\-(?:"
            + "|".join(escaped_genes)
            + r"|C\. elegans))[\s\.;:,'\"\)\]\}\>\?]"
        )
    else:
        anti_gene_pattern = (
            r"(?i)[\s\(\[\{\.,;:\'\"\<](anti\-C\. elegans)[\s\.;:,'\"\)\]\}\>\?]"
        )
    anti_gene_regex = re.compile(anti_gene_pattern)

    additional_keywords_regex = re.compile(
        r"[\s\(\[\{\.,;:\'\"\<]("
        + "|".join(re.escape(k) for k in ADDITIONAL_KEYWORDS)
        + r")[\s\.;:,'\"\)\]\}\>\?]"
    )

    combinations = list(itertools.product(COMBINATION_1, COMBINATION_2))
    combinations_regex = [
        (
            comb,
            re.compile(".*" + comb[0] + r"\s.*" + comb[1] + r"[\s\.;:,'\"\)\]\}\>\?]"),
            re.compile(".*" + comb[1] + r"\s.*" + comb[0] + r"[\s\.;:,'\"\)\]\}\>\?]"),
        )
        for comb in combinations
    ]

    return AntibodyRules(anti_gene_regex, additional_keywords_regex, combinations_regex)


def _normalize(sentence: str) -> str:
    """Replace various Unicode dashes with a plain hyphen so regexes hit them."""
    return sentence.replace('–', '-').replace('‐', '-')


def match_antibody_spans(sentences: Iterable[str], rules: AntibodyRules) -> set:
    """Apply all three rule sets to a list of sentences. Return the union of
    matched span strings.

    The case-sensitivity filter on anti-GENE matches requires at least one
    uppercase character in the gene-name suffix (so `anti-pdr-1` is dropped
    but `anti-PDR-1` and `anti-C. elegans` are kept).
    """
    matches: set = set()

    for raw in sentences:
        # Wrap with spaces so the leading/trailing char-class boundaries
        # in the regexes can match at sentence edges.
        sentence = " " + _normalize(raw) + " "

        # 1) anti-GENE patterns
        anti_gene_hits = rules.anti_gene_regex.findall(sentence)
        anti_gene_hits = [
            m for m in anti_gene_hits
            if m[5:].lower() != m[5:]
        ]
        matches.update(anti_gene_hits)

        # 2) Combination patterns
        sentence_lower = sentence.lower()
        for comb, regex_a_first, regex_b_first in rules.combinations_regex:
            if regex_a_first.match(sentence_lower) or regex_b_first.match(sentence_lower):
                matches.add(comb[0] + " " + comb[1])

        # 3) Additional keyword regex
        matches.update(rules.additional_keywords_regex.findall(sentence))

    return matches
