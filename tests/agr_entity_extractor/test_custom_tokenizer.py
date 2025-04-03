from agr_entity_extractor.models import CustomTokenizer


def test_extractor_case_sensitive_only():
    tokenizer = CustomTokenizer(tokens=["foo", "Bar-1"])
    text = "foo Foo and BAR-1"
    # Extract using the pattern "foo"
    results = tokenizer.tokenize(text)

    # Only the exact case match should be returned.
    assert results == text.split(" ")
    assert len([token_id for token_id in tokenizer(text)['input_ids'][0] if token_id != 0]) == 1


def test_extractor_case_sensitive_and_upper():
    tokenizer = CustomTokenizer(tokens=["foo", "Bar"], match_uppercase_entities=True)
    text = "foo Foo and BAR"
    # Extract using the pattern "foo"
    results = tokenizer.tokenize(text)

    # Only the exact case match should be returned.
    assert results == text.split(" ")
    assert len([token_id for token_id in tokenizer(text)['input_ids'][0] if token_id != 0]) == 2
