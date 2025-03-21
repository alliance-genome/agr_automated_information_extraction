import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, PretrainedConfig, AutoTokenizer, PreTrainedTokenizerFast
from agr_entity_extractor.models import AllianceStringMatchingEntityExtractor, CustomTokenizer


@pytest.fixture
def extractor():
    c_elegans_genes = ["lin-12", "ced-3"]
    config = PretrainedConfig(num_labels=2)

    def custom_tokenizer(doc):
        return doc.split()

    tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
    tfidf_vectorizer.fit(["dummy document with lin-12 and ced-3", "ced-3 is very popular", "ced-3 many times"])
    # Wrap the tokenizer with the custom pre-tokenizer
    return AllianceStringMatchingEntityExtractor(
        config=config,
        entities_to_extract=c_elegans_genes,
        min_matches=1,
        tfidf_threshold=0.1,
        match_uppercase=False,
        tokenizer=custom_tokenizer,
        vectorizer=tfidf_vectorizer
    )


def test_gene_name_extraction(extractor):
    text = ("This is an example text with some gene names like lin-12 and ced-3. Lin-12 is very important. "
            "I mean it, lin-12 is the bext.")
    nlp_pipeline = pipeline("ner", model=extractor, tokenizer=extractor.tokenizer)
    results = nlp_pipeline(text)
    extracted_entities = [result['word'] for result in results if result['entity'] == "LABEL_1"]
    assert "lin-12" in extracted_entities
    assert "ced-3" in extracted_entities


if __name__ == "__main__":
    pytest.main()
