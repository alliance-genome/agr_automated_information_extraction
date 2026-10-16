from agr_entity_extractor.models import CustomTokenizer


def test_extractor_case_sensitive_only():
    tokenizer = CustomTokenizer(tokens=["foo", "Bar-1"])
    text = "foo Foo and BAR-1"
    results = tokenizer.tokenize(text)

    # Only the curated codes are emitted, and only on an exact case match.
    assert results == ["foo"]
    assert len([token_id for token_id in tokenizer(text)['input_ids'][0] if token_id != 0]) == 1


def test_extractor_case_sensitive_and_upper():
    tokenizer = CustomTokenizer(tokens=["foo", "Bar"], match_uppercase_entities=True)
    text = "foo Foo and BAR"
    results = tokenizer.tokenize(text)

    # match_uppercase_entities compiles the pattern with re.IGNORECASE, so every case
    # variant of a curated code is emitted; non-entity words like "and" are not.
    assert results == ["foo", "Foo", "BAR"]
    # Only the exact curated spelling is in the vocab; the other variants map to [UNK] (0).
    assert len([token_id for token_id in tokenizer(text)['input_ids'][0] if token_id != 0]) == 1
