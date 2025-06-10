import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, PretrainedConfig
from agr_entity_extractor.models import (
    AllianceStringMatchingEntityExtractor,
    CustomTokenizer
)


# C. elegans entities extracted from realistic TEI files
C_ELEGANS_GENES = [
    'let-7', 'lin-41', 'mir-48', 'mir-84', 'mir-241', 'unc-54', 'dpy-30',
    'lin-29', 'hbl-1', 'zipt-7.1', 'spe-8', 'spe-6', 'spe-4', 'fem-3',
    'him-5', 'fog-2', 'glp-4', 'dpy-5', 'dpy-13', 'dpy-10', 'rol-6',
    'rrf-1', 'swm-1', 'try-5', 'ced-3', 'lin-12', 'fog-1', 'fem-1'
]

C_ELEGANS_STRAINS = [
    'N2', 'CB4856', 'JU1373', 'hc130', 'ok971', 'e1490', 'q96',
    'xe83', 'xe76', 'n2853', 'bn2', 'pk1417'
]


@pytest.fixture
def simple_extractor():
    """Simple extractor with original test genes for backward compatibility."""
    c_elegans_genes = ["lin-12", "ced-3"]
    config = PretrainedConfig(num_labels=2)
    custom_tokenizer = CustomTokenizer(tokens=c_elegans_genes)
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=lambda doc: custom_tokenizer.tokenize(doc)
    )
    training_docs = [
        "dummy document with lin-12 and ced-3",
        "ced-3 is very popular",
        "ced-3 many times"
    ]
    tfidf_vectorizer.fit(training_docs)
    return AllianceStringMatchingEntityExtractor(
        config=config,
        entities_to_extract=c_elegans_genes,
        min_matches=1,
        tfidf_threshold=0.1,
        match_uppercase=False,
        tokenizer=custom_tokenizer,
        vectorizer=tfidf_vectorizer
    )


@pytest.fixture
def realistic_extractor():
    """Extractor with realistic C. elegans entities from TEI files."""
    all_entities = C_ELEGANS_GENES + C_ELEGANS_STRAINS
    config = PretrainedConfig(num_labels=2)
    custom_tokenizer = CustomTokenizer(tokens=all_entities)

    # Training documents with realistic gene/strain mentions
    training_docs = [
        "The let-7 miRNA family regulates lin-41 in C. elegans development.",
        "zipt-7.1 controls sperm activation in strain N2 and CB4856.",
        "Mutations in spe-8, spe-6, and spe-4 affect sperm function.",
        "The him-5(e1490) strain produces more males.",
        "fog-2(q71) females are useful for crossing experiments.",
        "mir-48 and mir-84 are members of the let-7 family."
    ]

    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=lambda doc: custom_tokenizer.tokenize(doc)
    )
    tfidf_vectorizer.fit(training_docs)

    return AllianceStringMatchingEntityExtractor(
        config=config,
        entities_to_extract=all_entities,
        min_matches=1,
        tfidf_threshold=0.05,
        match_uppercase=False,
        tokenizer=custom_tokenizer,
        vectorizer=tfidf_vectorizer
    )


def test_tfidf_vectorization(simple_extractor):
    """Test TF-IDF vectorization with simple gene entities."""
    text = (
        "This is an example text with some gene names like lin-12 and ced-3. "
        "Lin-12 is very important. I mean it, lin-12 is the best."
    )
    tfidf_vectorizer = simple_extractor.vectorizer
    doc_tfidf = tfidf_vectorizer.transform([text])
    feature_index = tfidf_vectorizer.vocabulary_["lin-12"]
    tfidf_value = doc_tfidf[0, feature_index]
    assert tfidf_value > 0


def test_gene_name_extraction(simple_extractor):
    """Test basic gene name extraction with simple entities."""
    text = (
        "This is an example text with some gene names like lin-12 and ced-3. "
        "Lin-12 is very important. I mean it, lin-12 is the best."
    )
    nlp_pipeline = pipeline(
        "ner",
        model=simple_extractor,
        tokenizer=simple_extractor.tokenizer
    )
    results = nlp_pipeline(text)
    extracted_entities = [
        result['word'] for result in results
        if result['entity'] == "LABEL_1"
    ]
    assert "lin-12" in extracted_entities
    assert "ced-3" in extracted_entities


# Test data from realistic TEI files
MIRNA_PAPER_TEXT = """
An interplay of miRNA abundance and target site architecture determines 
miRNA activity and specificity. MicroRNAs often occur in families whose 
members share an identical 5 terminal 'seed' sequence. Using the let-7 
miRNA family in Caenorhabditis elegans, we find that seed match 
imperfections can increase specificity by requiring extensive pairing 
outside the miRNA seed region for efficient silencing. let-7 ensures 
proper development of C. elegans by repressing one crucial target, lin-41, 
whose 3' UTR contains two functional let-7 binding sites. Both sites 
contain imperfect seed-matches, which yield a bulged-out nucleotide and 
a G:U wobble base-pair respectively. Moreover, both sites exhibit extensive 
complementarity to the seed-distal sequence of let-7 but to none of its 
sisters mir-48, mir-84, and mir-241. The transgenic unc-54 + miRNA sites 
reporter strains were obtained by single-copy integration. Both reporters 
are driven by the ubiquitous and constitutively active dpy-30 promoter and 
contain the unc-54 3' UTR. Wild-type worms were injected with 
strain-specific constructs including dpy-10 co-crispr mix. The 
lin-41(xe83[perfect]) allele differs from the wildtype allele in two 
nucleotides.
"""

SPERM_PAPER_TEXT = """
The zinc transporter ZIPT-7.1 regulates sperm activation in nematodes. 
The zipt-7.1 mutant hermaphrodites cannot self-fertilize, and males 
reproduce poorly, because mutant spermatids are defective in responding 
to activating signals. C. elegans strains were derived from Bristol N2. 
They include fog-1(q253) I, glp-4(bn2) I, rrf-1(pk1417) I, 
Dsp2/spe-8(hc53) dpy-5 I, spe-6(hc163) dpy-18(e364) III, dpy-13(e184) IV, 
zipt-7.1(hc130) IV, zipt-7.1(ok971) IV, zipt-7.1(as42) IV, fem-3(q96) IV, 
fem-1(hc17) IV, swm-1(ok1193) V, him-5(e1490) V, and fog-2(q71) V. Dpy 
hermaphrodites from the strain hc130 dpy-13(e184)/nT1 were crossed with 
N2 males to separate the hc130 allele from dpy-13. Next, sterile, non-Dpy 
hermaphrodites isolated from the F2 were crossed with males from the 
polymorphic strain CB4856. The zipt-7.1 gene is expressed in the germ line 
and functions in germ cells to promote sperm activation. Genetic epistasis 
places zipt-7.1 at the end of the spe-8 sperm activation pathway, and 
ZIPT-7.1 binds SPE-4, a presenilin that regulates sperm activation.
"""

def test_realistic_mirna_entities_extraction(realistic_extractor):
    """Test entity extraction from realistic miRNA paper text."""
    nlp_pipeline = pipeline(
        "ner",
        model=realistic_extractor,
        tokenizer=realistic_extractor.tokenizer
    )
    results = nlp_pipeline(MIRNA_PAPER_TEXT)
    extracted_entities = [
        result['word'] for result in results
        if result['entity'] == "LABEL_1"
    ]

    # Should find miRNA family genes
    assert "let-7" in extracted_entities
    assert "lin-41" in extracted_entities
    mirna_family_found = any(
        gene in extracted_entities
        for gene in ["mir-48", "mir-84", "mir-241"]
    )
    assert mirna_family_found

    # Should find reporter genes
    assert "unc-54" in extracted_entities
    assert "dpy-30" in extracted_entities

    # Should find strain/allele identifiers
    assert "xe83" in extracted_entities
    assert "dpy-10" in extracted_entities

    print(f"miRNA paper entities found: {extracted_entities}")


def test_realistic_sperm_entities_extraction(realistic_extractor):
    """Test entity extraction from realistic sperm biology paper text."""
    nlp_pipeline = pipeline(
        "ner",
        model=realistic_extractor,
        tokenizer=realistic_extractor.tokenizer
    )
    results = nlp_pipeline(SPERM_PAPER_TEXT)
    extracted_entities = [
        result['word'] for result in results
        if result['entity'] == "LABEL_1"
    ]

    # Should find zinc transporter genes
    assert "zipt-7.1" in extracted_entities

    # Should find sperm-related genes
    assert "spe-8" in extracted_entities
    assert "spe-6" in extracted_entities

    # Should find other fertility genes
    fertility_genes_found = any(
        gene in extracted_entities
        for gene in ["fog-1", "fog-2"]
    )
    assert fertility_genes_found
    assert "fem-3" in extracted_entities
    assert "him-5" in extracted_entities

    # Should find strain names
    assert "N2" in extracted_entities
    assert "CB4856" in extracted_entities

    # Should find allele identifiers
    assert "hc130" in extracted_entities
    assert "ok971" in extracted_entities
    assert "e1490" in extracted_entities

    print(f"Sperm paper entities found: {extracted_entities}")


def test_realistic_tfidf_features(realistic_extractor):
    """Test TF-IDF vectorization with realistic gene/strain entities."""
    tfidf_vectorizer = realistic_extractor.vectorizer

    # Test document mentioning multiple entities
    test_text = (
        "The let-7 family includes mir-48 and mir-84, which regulate "
        "lin-41 in strain N2."
    )
    doc_tfidf = tfidf_vectorizer.transform([test_text])

    # Check that known entities have positive TF-IDF scores
    entities_to_check = ["let-7", "mir-48", "mir-84", "lin-41", "N2"]
    for entity in entities_to_check:
        if entity in tfidf_vectorizer.vocabulary_:
            feature_index = tfidf_vectorizer.vocabulary_[entity]
            tfidf_value = doc_tfidf[0, feature_index]
            assert tfidf_value > 0, (
                f"Entity '{entity}' should have positive TF-IDF score"
            )


def test_strain_and_gene_distinction(realistic_extractor):
    """Test that the extractor can identify both genes and strain names."""
    # Text with both genes and strains clearly mentioned
    mixed_text = (
        "We crossed him-5(e1490) males with fem-3(q96) hermaphrodites "
        "from strain CB4856 to study let-7 regulation."
    )

    nlp_pipeline = pipeline(
        "ner",
        model=realistic_extractor,
        tokenizer=realistic_extractor.tokenizer
    )
    results = nlp_pipeline(mixed_text)
    extracted_entities = [
        result['word'] for result in results
        if result['entity'] == "LABEL_1"
    ]

    # Should find genes
    assert "him-5" in extracted_entities
    assert "fem-3" in extracted_entities
    assert "let-7" in extracted_entities

    # Should find strain identifiers
    assert "e1490" in extracted_entities
    assert "q96" in extracted_entities
    assert "CB4856" in extracted_entities

    print(f"Mixed entities found: {extracted_entities}")


def test_comprehensive_entity_coverage(realistic_extractor):
    """Test extractor covers comprehensive range of C. elegans entities."""
    # Test text covering different entity types
    comprehensive_text = """
    In this study, we used C. elegans strain N2 and the Hawaiian 
    strain CB4856. We examined sperm genes including spe-8, spe-6, 
    spe-4, and zipt-7.1. Fertility genes fog-1, fog-2, fem-3, and 
    glp-4 were also analyzed. The let-7 miRNA family members mir-48, 
    mir-84, and mir-241 regulate lin-41. Reporter constructs used 
    unc-54 and dpy-30 promoters. Mutant alleles included him-5(e1490), 
    dpy-13(e184), and zipt-7.1(hc130).
    """

    nlp_pipeline = pipeline(
        "ner",
        model=realistic_extractor,
        tokenizer=realistic_extractor.tokenizer
    )
    results = nlp_pipeline(comprehensive_text)
    extracted_entities = [
        result['word'] for result in results
        if result['entity'] == "LABEL_1"
    ]

    # Count different categories of entities found
    gene_categories = {
        'sperm_genes': ['spe-8', 'spe-6', 'spe-4', 'zipt-7.1'],
        'fertility_genes': ['fog-1', 'fog-2', 'fem-3', 'glp-4'],
        'mirna_genes': ['let-7', 'mir-48', 'mir-84', 'mir-241', 'lin-41'],
        'reporter_genes': ['unc-54', 'dpy-30'],
        'strains': ['N2', 'CB4856'],
        'alleles': ['e1490', 'e184', 'hc130']
    }

    found_categories = {}
    for category, entities in gene_categories.items():
        found_in_category = [e for e in entities if e in extracted_entities]
        found_categories[category] = found_in_category
        assert len(found_in_category) > 0, (
            f"Should find at least one entity from {category}"
        )

    print(f"Comprehensive test results: {found_categories}")
    print(f"Total entities found: {len(extracted_entities)}")

    # Should find a substantial number of entities
    assert len(extracted_entities) >= 15, (
        "Should extract a substantial number of entities from "
        "comprehensive text"
    )


if __name__ == "__main__":
    pytest.main()
