"""Tests for the curator-requested ZFIN gene extraction refinements in
utils.entity_extraction_utils:

- restrict_markdown_to_results_methods: keep Results + Materials/Methods only.
- gene_has_standalone_mention / filter_construct_embedded_genes: drop genes that
  only ever appear inside constructs such as Tg(gene:reporter).
"""

from utils.entity_extraction_utils import (
    _classify_section_heading,
    restrict_markdown_to_results_methods,
    gene_has_standalone_mention,
    filter_construct_embedded_genes,
    is_false_positive_allele,
)


# --------------------------------------------------------------------- #
# Section heading classification                                        #
# --------------------------------------------------------------------- #
def test_classify_keep_results_and_methods():
    assert _classify_section_heading("Results") == "keep"
    assert _classify_section_heading("Materials and Methods") == "keep"
    assert _classify_section_heading("Methods") == "keep"
    assert _classify_section_heading("Experimental Section") == "keep"


def test_classify_keep_handles_numbering_and_case():
    assert _classify_section_heading("3. Results") == "keep"
    assert _classify_section_heading("2. Materials and Methods") == "keep"
    assert _classify_section_heading("MATERIALS AND METHODS") == "keep"


def test_classify_combined_results_and_discussion_is_kept():
    # KEEP is tested before DROP, so the results content is retained.
    assert _classify_section_heading("Results and Discussion") == "keep"


def test_classify_drop_intro_discussion_refs():
    assert _classify_section_heading("Introduction") == "drop"
    assert _classify_section_heading("1. Introduction") == "drop"
    assert _classify_section_heading("Discussion") == "drop"
    assert _classify_section_heading("References") == "drop"
    assert _classify_section_heading("Conclusions") == "drop"


def test_classify_neutral_unknown_heading():
    assert _classify_section_heading("Statistical analysis") == "neutral"


# --------------------------------------------------------------------- #
# restrict_markdown_to_results_methods                                  #
# --------------------------------------------------------------------- #
_MD = """# A zebrafish paper

## Abstract

abstract-gene mention here.

## Introduction

intro-gene should be skipped.

## Methods

### Fish husbandry

methods-gene-a described.

## Results

results-gene-b measured.

## Discussion

discussion-gene should be skipped.

## References

reference-gene should be skipped.
"""


def test_restrict_keeps_results_and_methods_only():
    kept = restrict_markdown_to_results_methods(_MD)
    assert kept is not None
    assert "methods-gene-a" in kept
    assert "results-gene-b" in kept
    # Neutral subsection under Methods inherits the KEEP decision.
    assert "Fish husbandry" not in kept  # heading lines themselves are dropped
    assert "intro-gene" not in kept
    assert "discussion-gene" not in kept
    assert "reference-gene" not in kept
    assert "abstract-gene" not in kept


def test_restrict_neutral_subsection_inherits_keep():
    md = (
        "## Methods\n\n"
        "### Imaging\n\n"
        "sub-methods-gene here.\n\n"
        "## Discussion\n\n"
        "drop-gene here.\n"
    )
    kept = restrict_markdown_to_results_methods(md)
    assert "sub-methods-gene" in kept
    assert "drop-gene" not in kept


def test_restrict_returns_none_when_no_results_methods():
    md = "## Introduction\n\nintro only.\n\n## Discussion\n\nmore text.\n"
    assert restrict_markdown_to_results_methods(md) is None


def test_restrict_returns_none_for_unstructured_text():
    assert restrict_markdown_to_results_methods("plain text, no headers") is None
    assert restrict_markdown_to_results_methods("") is None


# --------------------------------------------------------------------- #
# Construct-embedded gene filtering                                     #
# --------------------------------------------------------------------- #
def test_standalone_plain_mention_kept():
    assert gene_has_standalone_mention("levels of fn1a were high", "fn1a")


def test_standalone_gene_only_in_construct_rejected():
    assert not gene_has_standalone_mention("only in Tg(sox10:GFP) here", "sox10")


def test_standalone_gene_in_construct_and_alone_kept():
    text = "sox10 expression increased; Tg(sox10:GFP) injected"
    assert gene_has_standalone_mention(text, "sox10")


def test_standalone_promoter_fusion_rejected():
    assert not gene_has_standalone_mention("the sox10:EGFP fusion", "sox10")


def test_standalone_ion_notation_rejected():
    # The calcium ion Ca2+ must not count as a mention of the gene ca2.
    assert not gene_has_standalone_mention("influx of Ca2+ into the cell", "ca2")
    # Superscript-plus variant (U+207A).
    assert not gene_has_standalone_mention("cytosolic Ca2⁺ levels", "ca2")


def test_standalone_gene_ion_and_alone_kept():
    # Present as the ion AND as a standalone gene mention -> kept.
    text = "Ca2+ signalling; the ca2 gene was upregulated"
    assert gene_has_standalone_mention(text, "ca2")


def test_standalone_gene_plain_not_ion_kept():
    # A bare ca2 with no trailing '+' is a real gene mention.
    assert gene_has_standalone_mention("expression of ca2 increased", "ca2")


# --------------------------------------------------------------------- #
# Allele FP: Xenopus Nieuwkoop-Faber developmental-staging collisions   #
# --------------------------------------------------------------------- #
_STAGING_TEXT = (
    "Embryos were fixed at the following developmental stages: NF st9 (n = 6), "
    "st10.5 (n = 6), st12.5 (n = 6), st18 (n = 5), st20 (n = 5), st23 (n = 6), "
    "st28 (n = 6) and st40 (n = 5)."
)


def test_allele_staging_st_token_rejected_in_staging_context():
    for stg in ("st9", "st20", "st23"):
        is_fp, reason = is_false_positive_allele(_STAGING_TEXT, stg)
        assert is_fp, f"{stg} should be dropped in NF-staging context"
        assert "developmental stage" in reason


def test_allele_st_token_kept_without_staging_context():
    # Same st-token, ordinary allele paper (no NF/decimal-stage/Nieuwkoop cue).
    text = "the st20 mutant showed a fin phenotype; st20 carriers were crossed"
    is_fp, _ = is_false_positive_allele(text, "st20")
    assert not is_fp


def test_allele_non_st_token_unaffected_by_staging_guard():
    # A non-st allele in a paper that also uses staging notation is not touched
    # by the staging guard (may still pass/fail other rules; here it passes).
    is_fp, reason = is_false_positive_allele(_STAGING_TEXT, "vo84")
    assert not (is_fp and "developmental stage" in reason)


def test_standalone_zgc_id_with_internal_colon_kept():
    # The colon is INTERNAL to the name, not a construct delimiter.
    assert gene_has_standalone_mention("the zgc:174917 gene", "zgc:174917")


def test_standalone_substring_of_longer_identifier_rejected():
    assert not gene_has_standalone_mention("fn1ab is different", "fn1a")
    assert not gene_has_standalone_mention("nkx2.1a cells", "nkx2.1")


def test_standalone_at_text_edges_kept():
    assert gene_has_standalone_mention("sox10", "sox10")           # whole text
    assert gene_has_standalone_mention("we saw slc26a4.", "slc26a4")  # sentence end


def test_standalone_enclosed_in_parens_rejected():
    assert not gene_has_standalone_mention("(slc26a4)", "slc26a4")


def test_filter_construct_embedded_partitions():
    text = "sox10 is expressed. Tg(fn1a:GFP) was used."
    kept, dropped = filter_construct_embedded_genes(["sox10", "fn1a"], text)
    assert kept == ["sox10"]
    assert dropped == ["fn1a"]


def test_filter_construct_empty_text_keeps_all():
    kept, dropped = filter_construct_embedded_genes(["sox10", "fn1a"], "")
    assert kept == ["sox10", "fn1a"]
    assert dropped == []
