import pytest
from agr_entity_extractor.models import CustomTokenizer


class TestSubstringMatchingIssue:
    """Test cases for substring matching issues in CustomTokenizer"""
    
    def test_strain_name_substring_matching(self):
        """Test that longer strain names are matched before shorter substrings"""
        
        # C. elegans strain names where shorter names are substrings of longer ones
        # Intentionally provide them in shortest-first order to test sorting
        strain_names = [
            "N2L1",  # Longer strain name containing N2
            "N2",  # Short strain name
            "CB4856",  # Another short strain
            "CB4856dhIs4",  # Longer strain containing CB4856
            "test"
        ]
        
        tokenizer = CustomTokenizer(tokens=strain_names)
        
        # Test text containing both short and long strain names
        test_text = "We used strains N2L1 and CB4856dhIs4 in our experiments with and CB4856. testing"
        
        # Tokenize the text
        tokens = tokenizer._tokenize(test_text)
        
        # Check which tokens are recognized as entities (non-UNK)
        token_ids = tokenizer(test_text)['input_ids'][0]
        recognized_entities = []
        
        for token, token_id in zip(tokens, token_ids):
            if token_id != tokenizer.unk_token_id:  # Not UNK token
                recognized_entities.append(token)
        
        # All strain names should be recognized
        expected_entities = ["N2L1", "CB4856dhIs4", "CB4856"]
        assert set(recognized_entities) == set(expected_entities), \
            f"Expected {expected_entities}, but got {recognized_entities}"
        
        # Verify that longer strain names are not broken into shorter substrings
        assert "N2L1" in recognized_entities, "N2L1 should be recognized as a single entity"
        assert "CB4856dhIs4" in recognized_entities, "CB4856dhIs4 should be recognized as a single entity"
        assert "N2" not in recognized_entities
        assert "test" not in recognized_entities
    
    def test_gene_name_substring_matching(self):
        """Test substring matching with C. elegans gene names"""
        
        gene_names = [
            "lin-12",
            "lin-12A",  # Longer gene name containing lin-12
            "ced-3",
            "ced-3p",   # Longer gene name containing ced-3
        ]
        
        tokenizer = CustomTokenizer(tokens=gene_names)
        
        test_text = "The genes lin-12A and ced-3p are related to lin-12 and ced-3 respectively."
        
        tokens = tokenizer._tokenize(test_text)
        token_ids = tokenizer(test_text)['input_ids'][0]
        recognized_entities = []
        
        for token, token_id in zip(tokens, token_ids):
            if token_id != tokenizer.unk_token_id:
                recognized_entities.append(token)
        
        expected_entities = ["lin-12A", "ced-3p", "lin-12", "ced-3"]
        assert set(recognized_entities) == set(expected_entities), \
            f"Expected {expected_entities}, but got {recognized_entities}"
    
    def test_regex_pattern_ordering(self):
        """Test that the regex pattern orders entities by length (longest first)"""
        
        entities = ["test", "testing", "tested", "te"]
        tokenizer = CustomTokenizer(tokens=entities)
        
        # Extract the entity pattern from the compiled regex
        pattern_str = tokenizer.pattern.pattern
        
        # The pattern should have entities ordered by length (longest first)
        assert "testing" in pattern_str
        assert "tested" in pattern_str  
        assert "test" in pattern_str
        assert "te" in pattern_str
        
        # Check that the alternation order is longest first
        # The pattern should be like: (testing|tested|test|te)|...
        entity_part = pattern_str.split('|')[0].strip('(')  # Get the entity alternation part
        
        # Split by | and check order
        alternatives = entity_part.split('|')
        if len(alternatives) >= 3:  # Ensure we have enough alternatives to test
            # The entities should be ordered by length, longest first
            lengths = [len(alt) for alt in alternatives[:4]]  # Check first 4
            assert lengths == sorted(lengths, reverse=True), \
                f"Entities should be ordered by length (longest first), got: {alternatives[:4]}"
    
    def test_overlapping_entities_in_text(self):
        """Test behavior when entities overlap in text"""
        
        entities = ["test", "testing", "tested"]
        tokenizer = CustomTokenizer(tokens=entities)
        
        # Text where entities could overlap
        test_text = "We are testing and tested our test cases."
        
        tokens = tokenizer._tokenize(test_text)
        token_ids = tokenizer(test_text)['input_ids'][0]
        recognized_entities = []
        
        for token, token_id in zip(tokens, token_ids):
            if token_id != tokenizer.unk_token_id:
                recognized_entities.append(token)
        
        # Should recognize the longest possible matches
        expected_entities = ["testing", "tested", "test"]
        assert set(recognized_entities) == set(expected_entities), \
            f"Expected {expected_entities}, but got {recognized_entities}"
    
    def test_shortest_first_input_order(self):
        """Test that tokenizer works correctly even when entities are provided shortest-first"""
        
        # Provide entities in deliberately problematic order (shortest first)
        entities = ["N2", "OP50", "N2L1", "OP50-1", "CB4856", "CB4856dhIs4"]
        tokenizer = CustomTokenizer(tokens=entities)
        
        # Text designed to trigger substring issues if sorting fails
        test_text = "Strains N2L1, CB4856dhIs4, and OP50-1 compared to N2, CB4856, and OP50."
        
        tokens = tokenizer._tokenize(test_text)
        token_ids = tokenizer(test_text)['input_ids'][0]
        recognized_entities = []
        
        for token, token_id in zip(tokens, token_ids):
            if token_id != tokenizer.unk_token_id:
                recognized_entities.append(token)
        
        # Should still get longest matches first, not substrings
        expected_entities = ["N2L1", "CB4856dhIs4", "OP50-1", "N2", "CB4856", "OP50"]
        assert set(recognized_entities) == set(expected_entities), \
            f"Expected {expected_entities}, but got {recognized_entities}"
        
        # Critically important: verify no substring breaking occurred
        assert "N2L1" in recognized_entities, "N2L1 should not be broken into N2 + L1"
        assert "CB4856dhIs4" in recognized_entities, "CB4856dhIs4 should not be broken into CB4856 + dhIs4"
        assert "OP50-1" in recognized_entities, "OP50-1 should not be broken into OP50 + -1"
    
    def test_full_alliance_extractor_pipeline(self):
        """Test the complete AllianceStringMatchingEntityExtractor pipeline for substring issues"""
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from transformers import PretrainedConfig
        from agr_entity_extractor.models import AllianceStringMatchingEntityExtractor
        
        # Problematic entity set (shortest first to stress test)
        strain_names = ["N2", "CB4856", "OP50", "N2L1", "CB4856dhIs4", "OP50-1"]
        
        config = PretrainedConfig(num_labels=2)
        custom_tokenizer = CustomTokenizer(tokens=strain_names)
        
        # Fit TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: custom_tokenizer._tokenize(doc))
        dummy_docs = [
            "N2L1 strain with CB4856dhIs4 and OP50-1",
            "N2 control with CB4856 and OP50",
            "strain N2L1 CB4856dhIs4 OP50-1 experiments"
        ]
        tfidf_vectorizer.fit(dummy_docs)
        
        # Create the full extractor
        extractor = AllianceStringMatchingEntityExtractor(
            config=config,
            entities_to_extract=strain_names,
            min_matches=1,
            tfidf_threshold=0.0,  # Set to 0 to bypass TF-IDF filtering
            match_uppercase=False,
            tokenizer=custom_tokenizer,
            vectorizer=tfidf_vectorizer
        )
        
        # Test text with potential substring conflicts
        test_text = "Using N2L1, CB4856dhIs4, and OP50-1 strains versus N2, CB4856, and OP50 controls."
        
        # Get tokenizer output
        tokenizer_output = custom_tokenizer(test_text, return_tensors="pt")
        input_ids = tokenizer_output["input_ids"]
        
        # Run the full pipeline
        model_output = extractor.forward(input_ids)
        logits = model_output["logits"]
        
        # Extract predicted entities
        predictions = (logits[:, :, 1] > 0.5).squeeze().tolist()
        tokens = custom_tokenizer._tokenize(test_text)
        
        predicted_entities = []
        for i, (token, is_entity) in enumerate(zip(tokens, predictions)):
            if is_entity:
                predicted_entities.append(token)
        
        print(f"Input text: {test_text}")
        print(f"Tokens: {tokens}")
        print(f"Predictions: {predictions}")
        print(f"Predicted entities: {predicted_entities}")
        
        # Verify that longer entities are correctly identified
        expected_entities = ["N2L1", "CB4856dhIs4", "OP50-1", "N2", "CB4856", "OP50"]
        assert set(predicted_entities) == set(expected_entities), \
            f"Expected {expected_entities}, but got {predicted_entities}"
        
        # Critical test: ensure no substring breaking
        assert "N2L1" in predicted_entities, "N2L1 should be detected as entity, not broken into substrings"
        assert "CB4856dhIs4" in predicted_entities, "CB4856dhIs4 should be detected as entity"
        assert "OP50-1" in predicted_entities, "OP50-1 should be detected as entity"
    
    def test_punctuation_boundary_matching(self):
        """Test that strain names are correctly matched when surrounded by various punctuation"""
        
        # Include both problematic substring pairs and regular strain names
        strain_names = ["N2", "N2L1", "CB4856", "CB4856dhIs4", "test", "testing"]
        tokenizer = CustomTokenizer(tokens=strain_names)
        
        # Test text with various punctuation scenarios
        test_cases = [
            # Round brackets - "testing" should be matched as separate entity
            ("Used strains (N2) and (CB4856dhIs4) for testing.", ["N2", "CB4856dhIs4", "testing"]),
            # Square brackets  
            ("Strains [N2L1] and [CB4856] were analyzed.", ["N2L1", "CB4856"]),
            # Followed by period
            ("The strain N2. was used as control.", ["N2"]),
            # Followed by comma
            ("Strains N2L1, CB4856dhIs4, and others.", ["N2L1", "CB4856dhIs4"]),
            # Preceded by dash - "Non-N2" should match N2 since it has proper boundaries
            ("Non-N2 strain was used.", ["N2"]),
            # Space-separated dash - this SHOULD match N2
            ("Non - N2 strain was used.", ["N2"]),
            # Mixed punctuation
            ("Results: (N2L1), [CB4856], test.", ["N2L1", "CB4856", "test"]),
            # Ensure "test" is NOT matched within "testing"
            ("We were testing the strain N2.", ["testing", "N2"]),
            # Multiple punctuation
            ("Strain (N2L1); CB4856dhIs4, test: results.", ["N2L1", "CB4856dhIs4", "test"]),
            # Beginning of sentence
            ("N2 strain showed results.", ["N2"]),
            # End of sentence with period
            ("We used CB4856dhIs4.", ["CB4856dhIs4"]),
            # Colon and semicolon
            ("Strains: N2L1; CB4856dhIs4:", ["N2L1", "CB4856dhIs4"]),
        ]
        
        for i, (test_text, expected_entities) in enumerate(test_cases):
            tokens = tokenizer._tokenize(test_text)
            token_ids = tokenizer(test_text)['input_ids'][0]
            
            recognized_entities = []
            for token, token_id in zip(tokens, token_ids):
                if token_id != tokenizer.unk_token_id:
                    recognized_entities.append(token)
            
            print(f"\nTest case {i+1}: {test_text}")
            print(f"Tokens: {tokens}")
            print(f"Expected: {expected_entities}")
            print(f"Recognized: {recognized_entities}")
            
            # Check that all expected entities are found
            for expected in expected_entities:
                assert expected in recognized_entities, \
                    f"Expected entity '{expected}' not found in {recognized_entities} for text: {test_text}"
            
            # Critical: ensure "test" is NOT found within "testing"
            if "testing" in test_text and "test" not in expected_entities:
                assert "test" not in recognized_entities, \
                    f"'test' should not be matched within 'testing' for text: {test_text}"
    
    def test_word_boundary_edge_cases(self):
        """Test edge cases for word boundary detection"""
        
        strain_names = ["N2", "test", "CB4856"]
        tokenizer = CustomTokenizer(tokens=strain_names)
        
        edge_cases = [
            # Should NOT match (substring cases)
            ("testing123", []),  # "test" within "testing123"
            ("preN2post", []),   # "N2" within "preN2post"  
            ("CB4856extra", []), # "CB4856" within "CB4856extra"
            
            # Should match (proper boundaries)
            ("test-case", ["test"]),      # "test" before hyphen
            ("N2-strain", ["N2"]),        # "N2" before hyphen  
            ("CB4856_variant", ["CB4856"]), # "CB4856" before underscore
            ("strain:N2", ["N2"]),        # "N2" after colon
            ("(test)", ["test"]),         # "test" in parentheses
            ("'N2'", ["N2"]),            # "N2" in quotes
            ("\"CB4856\"", ["CB4856"]),   # "CB4856" in double quotes
        ]
        
        for i, (test_text, expected_entities) in enumerate(edge_cases):
            tokens = tokenizer._tokenize(test_text)
            token_ids = tokenizer(test_text)['input_ids'][0]
            
            recognized_entities = []
            for token, token_id in zip(tokens, token_ids):
                if token_id != tokenizer.unk_token_id:
                    recognized_entities.append(token)
            
            print(f"\nEdge case {i+1} - Text: {test_text}")
            print(f"Expected: {expected_entities}")
            print(f"Recognized: {recognized_entities}")
            
            assert set(recognized_entities) == set(expected_entities), \
                f"Expected {expected_entities}, got {recognized_entities} for text: {test_text}"

