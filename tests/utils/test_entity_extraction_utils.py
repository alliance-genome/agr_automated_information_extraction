"""Tests for the curator-requested ZFIN gene extraction refinements in
utils.entity_extraction_utils:

- restrict_markdown_to_results_methods: keep Results + Materials/Methods only.
- gene_has_standalone_mention / filter_construct_embedded_genes: drop genes that
  only ever appear inside constructs such as Tg(gene:reporter), hyphenated
  foreign gene names (sel-12), URLs / domains (zfin.org) or subscripts
  (k<sub>cat</sub>).
- filter_english_word_gene_symbols: English-word symbols (nor, top) need
  italic gene typography before a paper is credited with them.
"""

from utils.entity_extraction_utils import (
    _classify_section_heading,
    restrict_markdown_to_results_methods,
    gene_has_standalone_mention,
    filter_construct_embedded_genes,
    filter_english_word_gene_symbols,
    strip_non_gene_spans,
    is_false_positive_allele,
    rescue_zfin_all_letter_genes_from_markdown,
    _inside_open_parenthetical,
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
    # non-ASCII letters are identifier characters too (author name Rösel)
    assert not gene_has_standalone_mention("Rösel et al. 2011; Rösel TD, Hung L-H", "sel")
    assert not gene_has_standalone_mention("the Müller glia", "ller")


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


# --------------------------------------------------------------------- #
# Curator round 2: sel-12, URLs, subscripts, English-word symbols       #
# --------------------------------------------------------------------- #
def test_standalone_hyphen_digit_gene_name_rejected():
    # C. elegans sel-12 must not count as a mention of the ZFIN gene sel.
    assert not gene_has_standalone_mention("the presenilin sel-12 mutant", "sel")
    assert not gene_has_standalone_mention("let-7 miRNA", "let")


def test_standalone_hyphen_then_letter_still_kept():
    # Only hyphen+digit is a continuation; "sox10-positive" is a real mention.
    assert gene_has_standalone_mention("sox10-positive cells", "sox10")


def test_standalone_gene_hyphenated_and_alone_kept():
    assert gene_has_standalone_mention("sel-12 in worms; zebrafish sel was cloned", "sel")


def test_standalone_bare_domain_rejected():
    # ".org" in zfin.org is not the ZFIN gene org.
    assert not gene_has_standalone_mention("see zfin.org for details", "org")
    assert not gene_has_standalone_mention("the top.png file", "top")


def test_standalone_sentence_end_dot_still_kept():
    # A sentence-final dot followed by whitespace is punctuation, not a domain.
    assert gene_has_standalone_mention("we studied org. Next we", "org")
    assert gene_has_standalone_mention("we saw slc26a4.", "slc26a4")


def test_standalone_scheme_url_rejected():
    # Path segments of a URL are not gene mentions either.
    assert not gene_has_standalone_mention("https://zfin.org/action/top/view", "top")
    assert not gene_has_standalone_mention("at http://example.org/cat here", "cat")
    assert not gene_has_standalone_mention("mail cat@zfin.org now", "cat")


def test_standalone_url_and_alone_kept():
    assert gene_has_standalone_mention("https://zfin.org/org ; the org gene", "org")


def test_standalone_subscript_rejected():
    # k<sub>cat</sub>/K<sub>m</sub> is enzyme kinetics, not the catalase gene cat.
    assert not gene_has_standalone_mention("the k<sub>cat</sub>/K<sub>m</sub> values", "cat")
    assert not gene_has_standalone_mention("k<SUB>cat</SUB> for caspase", "cat")
    # pandoc-style subscript
    assert not gene_has_standalone_mention("the k~cat~ values", "cat")
    # unclosed tag, as the curator quoted it ("k<sub>cat")
    assert not gene_has_standalone_mention("using k<sub>cat", "cat")
    assert not gene_has_standalone_mention("k<sub>cat /K<sub>m values", "cat")


def test_standalone_flattened_kinetic_constant_rejected():
    # OCR/PDF extraction flattens k<sub>cat</sub> into "k cat".
    assert not gene_has_standalone_mention("values for k cat \\K m were estimated", "cat")
    assert not gene_has_standalone_mention("the K cat of the enzyme", "cat")
    # an ordinary word ending in k before the gene is fine
    assert gene_has_standalone_mention("we knock cat down", "cat")
    # OCR glued the k onto the previous word; the trailing K m still marks kinetics
    assert not gene_has_standalone_mention("shown.Proteasek cat /K m (M -1", "cat")
    assert not gene_has_standalone_mention("Estimated k cat \\K m values", "cat")
    assert not gene_has_standalone_mention("E F k cat K m : [E]t", "cat")


def test_standalone_greek_letter_hyphen_prefix_rejected():
    # beta-catenin abbreviated as β-cat is not the catalase gene.
    assert not gene_has_standalone_mention("the mechanosensitive β-cat pathway", "cat")
    assert not gene_has_standalone_mention("Y667-β-cat site", "cat")
    # a hyphen after a full word is ordinary prose
    assert gene_has_standalone_mention("the anti-cat antibody", "cat")
    # Latin single-letter prefixes are real gene names and must be kept
    assert gene_has_standalone_mention("the proto-oncogene c-myb during", "myb")
    assert gene_has_standalone_mention("c-fos induction", "fos")


def test_standalone_catalog_number_rejected():
    assert not gene_has_standalone_mention("Invitrogen (cat. #C10640, lot #19)", "cat")
    assert not gene_has_standalone_mention("Sigma, cat. no. A1234", "cat")
    assert not gene_has_standalone_mention("Abcam cat# ab1234", "cat")
    # sentence-final "cat." followed by a new sentence is still a mention
    assert gene_has_standalone_mention("we measured cat. Next, sod rose", "cat")


def test_standalone_subscript_and_alone_kept():
    assert gene_has_standalone_mention("k<sub>cat</sub> values; cat expression rose", "cat")


def test_standalone_superscript_gene_untouched():
    # Superscripts carry alleles; the gene outside the tag is still a mention.
    assert gene_has_standalone_mention("nkx3.1<sup>ca116</sup> larvae", "nkx3.1")


def test_strip_non_gene_spans():
    assert strip_non_gene_spans("") == ""
    out = strip_non_gene_spans("go to https://zfin.org/a and k<sub>cat</sub> x")
    assert "zfin.org" not in out and "<sub>" not in out
    assert out.split() == ["go", "to", "and", "k", "x"]


def test_english_word_symbols_dropped_without_typography():
    kept, dropped = filter_english_word_gene_symbols(["nor", "top", "sox10"], [])
    assert kept == ["sox10"]
    assert dropped == ["nor", "top"]


def test_english_word_symbols_kept_with_italic_rescue():
    kept, dropped = filter_english_word_gene_symbols(["nor", "top", "sox10"], ["nor"])
    assert kept == ["nor", "sox10"]
    assert dropped == ["top"]


# --------------------------------------------------------------------- #
# Italic rescue: figure-panel labels and italicised subscripts          #
# --------------------------------------------------------------------- #
class _ZfinGeneModel:
    mod_abbr = "ZFIN"
    topic = "ATP:0000005"

    def __init__(self, *symbols):
        self.upper_to_original_mapping = {s.upper(): s for s in symbols}


def test_inside_open_parenthetical():
    text = "kinase A (PKA), (Fig. 5A, *top*) [11]"
    assert _inside_open_parenthetical(text, text.index("*top*"))
    text = "rows (*top*, *right arrowheads* and *bottom*, arrows)"
    assert _inside_open_parenthetical(text, text.index("*bottom*"))
    # closed parenthetical before the span -> not inside
    text = "(Fig. 1) shows *top* expression"
    assert not _inside_open_parenthetical(text, text.index("*top*"))
    # a closed nested group (markdown link target) does not end the outer one
    text = "stomach ([Fig. 1](#fig-1) *c*, *top*). At"
    assert _inside_open_parenthetical(text, text.index("*top*"))


def test_italic_rescue_skips_figure_panel_label():
    model = _ZfinGeneModel("top", "nor")
    md = "## Results\n\nPKA phosphorylation (Fig. 5A, *top*) [11] and (Fig. 6A, *top*) [11]."
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == []
    md = "## Results\n\nspots in bilateral rows (*top*, *right arrowheads* and *bottom*, arrows)."
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == []
    md = "## Results\n\nstomach ([Fig. 1](#fig-1) *c*, *top*). At 490 nm"
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == []


def test_italic_rescue_skips_construct_colon_and_partial_emphasis():
    model = _ZfinGeneModel("top", "cat", "org")
    md = "## Results\n\nin the *Tg* (*top*: dGFP) embryos; predicted with the *cat*RAPID server"
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == []


def test_italic_rescue_keeps_space_inside_italics():
    # pandoc puts the separating space inside the asterisks; still a whole word.
    model = _ZfinGeneModel("th", "gsc", "rho", "tnfa")
    md = ("## Results\n\nWe analyzed the *th *expression levels; the dorsal marker *gsc *and "
          "*eve1*; markers *il1b*, *mmp9 *and *tnfa *were chosen; and* rho* (NM_131084.1)")
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == ["gsc", "rho", "th", "tnfa"]


def test_italic_rescue_construct_only_dropped_but_mixed_kept():
    model = _ZfinGeneModel("kdrl", "mpx", "lck")
    # only ever inside Tg(gene:reporter) -> dropped, like the regex-path construct filter
    md = "## Results\n\nTG(*kdrl*:G-RCFP) fish were used; Tg(*kdrl*:G-RCFP) line was imaged."
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == []
    # construct plus a plain standalone mention elsewhere -> kept
    md = "## Results\n\nwe used Tg(*mpx*:GFP) fish; mpx-positive neutrophils were counted."
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == ["mpx"]
    # double-colon variant
    md = "## Results\n\nembryos from *lck*::GFP zebrafish were injected"
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == []


def test_italic_rescue_skips_segmented_italic_url():
    model = _ZfinGeneModel("org")
    md = "## Results\n\ndata from the *singlecell*.*broadinstitute*.*org* portal"
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == []
    md = "## Results\n\novarian markers (*zp2*, *org*, *sycp3*) confirmed"
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == ["org"]


def test_italic_rescue_keeps_gene_list_inside_figure_reference():
    # A figure reference that names the genes shown in the panel is real typography.
    # (digit-bearing symbols such as ptgs2b come from the body regex, not this rescue)
    model = _ZfinGeneModel("flnca", "ptgs2b", "th", "top")
    md = ("## Results\n\ngenes involved in regeneration (Fig. 6b, *flnca*, *ptgs2b*) "
          "and DA neurons (Supplementary Fig. 2d, *th*<sup>+</sup>, *top*).")
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == ["flnca", "th"]


def test_italic_rescue_keeps_italic_gene_outside_figure_reference():
    model = _ZfinGeneModel("top", "nor")
    md = "## Results\n\nExpression of *nor* was reduced (Fig. 2A)."
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == ["nor"]


def test_italic_rescue_skips_italic_subscript():
    model = _ZfinGeneModel("cat")
    md = "## Results\n\nthe *k*<sub>*cat*</sub> value and *k* <sub>*cat*</sub>/K<sub>m</sub> ratio"
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == []


def test_italic_rescue_keeps_italic_gene_next_to_subscript():
    model = _ZfinGeneModel("cat")
    md = "## Results\n\n*cat* mRNA rose; the *k*<sub>*cat*</sub> value was unchanged"
    assert rescue_zfin_all_letter_genes_from_markdown(md, model) == ["cat"]
