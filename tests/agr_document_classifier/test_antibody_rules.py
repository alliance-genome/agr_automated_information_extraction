from agr_document_classifier.antibody_rules import build_regexes, match_antibody_spans


def _matches(sentences, curated_genes):
    rules = build_regexes(curated_gene_names=curated_genes)
    return match_antibody_spans(sentences, rules)


def test_anti_gene_with_capitalized_suffix_matches():
    out = _matches(["This study used anti-PDR-1 antibody."], ["pdr-1"])
    assert "anti-PDR-1" in out


def test_anti_gene_lowercase_suffix_filtered_out():
    """Caltech filter keeps only matches where the gene-name suffix has at
    least one capital letter. anti-pdr-1 in lowercase is dropped."""
    out = _matches(["See anti-pdr-1 protein expression."], ["pdr-1"])
    assert not any(m.startswith("anti-") for m in out)


def test_excluded_gene_pdi_never_matches():
    """PDI is in EXCLUDE_GENES; even if passed in, it should not appear in
    the anti-GENE regex alternatives."""
    out = _matches(["See anti-PDI antibody and anti-Pdi protein."], ["pdi"])
    assert not any("pdi" in m.lower() for m in out)


def test_anti_c_elegans_matches():
    out = _matches(["The anti-C. elegans serum was used."], [])
    assert "anti-C. elegans" in out


def test_anti_msp_matches_via_additional_anti_keyword():
    """MSP is in ADDITIONAL_ANTI_KEYWORDS; antibody_rules adds it to the
    anti-GENE alternatives."""
    out = _matches(["The anti-MSP antibody recognized the protein."], [])
    assert "anti-MSP" in out


def test_combination_raised_antibody():
    out = _matches(["The antibody was raised against UNC-54."], ["unc-54"])
    assert "raised antibody" in out


def test_combination_preparation_antibodies_either_order():
    """Combinations match in either order: '<comb1> ... <comb2>' or
    '<comb2> ... <comb1>'."""
    out = _matches(["preparation of monoclonal antibodies for the study."], [])
    assert "preparation antibodies" in out


def test_additional_keyword_mh46_matches():
    out = _matches(["The MH46 antibody was used in this study."], [])
    assert "MH46" in out


def test_additional_keyword_sp56_matches():
    out = _matches(["Loaded the a-SP56 reagent into the gel."], [])
    assert "a-SP56" in out


def test_no_rule_fires_no_match():
    out = _matches(["We bought a commercial antibody from Sigma."], ["unc-54"])
    assert out == set()


def test_en_dash_normalized_to_hyphen():
    """'anti–PDR-1' (en-dash) should be treated as 'anti-PDR-1'."""
    out = _matches(["Used the anti–PDR-1 antibody."], ["pdr-1"])
    assert "anti-PDR-1" in out


def test_multiple_matches_in_one_sentence():
    out = _matches(
        ["The anti-PDR-1 antibody was raised in rabbits."],
        ["pdr-1"],
    )
    assert "anti-PDR-1" in out
    assert "raised antibody" in out
