import pytest
from transformers import pipeline, PretrainedConfig, AutoTokenizer
from agr_entity_extractor.models import AllianceStringMatchingEntityExtractor


@pytest.fixture
def extractor():
    c_elegans_genes = ["lin-12", "ced-3"]
    config = PretrainedConfig(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    tokenizer.add_tokens(c_elegans_genes)
    # Wrap the tokenizer with the custom pre-tokenizer
    return AllianceStringMatchingEntityExtractor(
        config=config,
        entities_to_extract=c_elegans_genes,
        min_matches=1,
        tfidf_threshold=0.1,
        match_uppercase=False,
        tokenizer=tokenizer,
        vectorizer=None
    )


def test_gene_name_extraction(extractor):
    text = "This is an example text with some gene names like lin-12 and ced-3."
    nlp_pipeline = pipeline("ner", model=extractor, tokenizer=extractor.tokenizer)
    results = nlp_pipeline(text)
    extracted_entities = [result['word'] for result in results if result['entity'] == "LABEL_1"]
    assert "lin-12" in extracted_entities
    assert "ced-3" in extracted_entities


if __name__ == "__main__":
    pytest.main()
