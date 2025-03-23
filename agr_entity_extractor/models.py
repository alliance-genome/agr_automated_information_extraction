import re
from collections import defaultdict
from typing import Dict

import torch
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, PreTrainedTokenizer


def convert_tokens_to_list_of_words(tokens):
    words = []
    current_word = ""

    for token in tokens:
        if token.startswith("##"):  # It's a subword, append to the current word
            current_word += token[2:]
        else:
            if current_word:  # Add the previous word to the list
                words.append(current_word)
            current_word = token  # Start a new word

    # Append the last word
    if current_word:
        words.append(current_word)

    return words


class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, tokens, match_uppercase_entities: bool = False, **kwargs):
        # Build your regex pattern (similar to before)
        self.pattern = None
        self.tokens = tokens
        self.match_uppercase_entities = match_uppercase_entities
        self.unk_token = "[UNK]"
        self.vocab = {self.unk_token: 0}
        self.add_tokens(tokens)
        self.update_vocab(tokens)
        # Set explicit unknown token id.
        self.unk_token_id = self.vocab[self.unk_token]
        # Some parts of the pipeline might look for `unknown_token_id`
        self.unknown_token_id = self.unk_token_id
        # Create a reverse mapping for converting IDs back to tokens.
        self.id_to_token = {id_: token for token, id_ in self.vocab.items()}
        self.model_max_length = int(1e30)
        super().__init__(**kwargs)

    def update_vocab(self, tokens):
        self.vocab = {self.unk_token: 0}
        for idx, token in enumerate(sorted(tokens), start=1):
            self.vocab[token] = idx
        # Set explicit unknown token id.
        self.unk_token_id = self.vocab[self.unk_token]
        # Some parts of the pipeline might look for `unknown_token_id`
        self.unknown_token_id = self.unk_token_id

    def add_tokens(self, new_tokens):
        self.tokens = list(set(self.tokens + new_tokens))
        if self.match_uppercase_entities:
            self.tokens = [token.upper() for token in self.tokens]
        escaped_entities = [re.escape(entity) for entity in sorted(self.tokens, key=len, reverse=True)]
        pattern_entities = "|".join(escaped_entities)
        fallback_pattern = r"\b\w+(?:[-']\w+)*\b"
        self.pattern = re.compile(f"({pattern_entities})|({fallback_pattern})", flags=re.IGNORECASE)
        self.tokens = list(set(self.tokens + new_tokens))
        self.update_vocab(new_tokens)

    def _tokenize(self, text, *args, **kwargs):
        # Use the simple tokenizer logic.
        if self.match_uppercase_entities:
            text = text.upper()
        tokens = []
        for match in re.finditer(self.pattern, text):
            token = match.group(1) or match.group(2)
            tokens.append(token)
        return tokens

    def __call__(self, text, *args, **kwargs):
        tokens = self._tokenize(text)
        # Convert tokens to numeric IDs using our simple vocab
        input_ids = [self.vocab.get(token, self.unknown_token_id) for token in tokens]

        # Build the output dictionary. Here we include "tokens" just for reference.
        output = {"input_ids": input_ids}

        # If the caller requests a special tokens mask, generate one.
        if kwargs.get("return_special_tokens_mask", False):
            # In this simple setup, none of our tokens are "special"
            special_tokens_mask = [0] * len(input_ids)
            output["special_tokens_mask"] = special_tokens_mask

        return_tensors = kwargs.get("return_tensors", None)
        if return_tensors == "pt":
            output["input_ids"] = torch.tensor([input_ids])
            if "special_tokens_mask" in output:
                output["special_tokens_mask"] = torch.tensor([output["special_tokens_mask"]])
            output["attention_mask"] = torch.ones_like(output["input_ids"])
        elif return_tensors == "tf":
            import tensorflow as tf
            output["input_ids"] = tf.convert_to_tensor([input_ids])
            if "special_tokens_mask" in output:
                output["special_tokens_mask"] = tf.convert_to_tensor([output["special_tokens_mask"]])
            output["attention_mask"] = tf.ones_like(output["input_ids"])
        else:
            # Wrap in a list to simulate a batch dimension.
            output["input_ids"] = [input_ids]
            if "special_tokens_mask" in output:
                output["special_tokens_mask"] = [output["special_tokens_mask"]]

        return output

    def get_vocab(self) -> Dict[str, int]:
        # This implementation returns the vocabulary dictionary.
        return self.vocab

    def convert_tokens_to_ids(self, tokens):
        """Converts tokens or a list of tokens to their corresponding IDs."""
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def _convert_id_to_token(self, index: int) -> str:
        """Converts a token ID back to the corresponding token string."""
        return self.id_to_token.get(index, self.unk_token)


class AllianceStringMatchingEntityExtractorConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "alliance_string_matching_entity_extractor"
        # Configure your labels as needed.
        self.id2label = {0: "O", 1: "ENTITY"}
        self.label2id = {"O": 0, "ENTITY": 1}


# Custom model for token classification using string matching.
class AllianceStringMatchingEntityExtractor(PreTrainedModel):
    config_class = AllianceStringMatchingEntityExtractorConfig

    def __init__(self, config, min_matches, tfidf_threshold,
                 tokenizer, vectorizer, entities_to_extract, load_entities_dynamically_fnc=None,
                 match_uppercase: bool = False):
        super().__init__(config)
        self.config = config
        self.tfidf_threshold = tfidf_threshold
        self.match_uppercase = match_uppercase
        self.min_matches = min_matches
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.entities_to_extract = set(entities_to_extract) if entities_to_extract else None
        self.load_entities_dynamically_fnc = load_entities_dynamically_fnc
        self.alliance_entities_loaded = False
        self.name_to_curie_mapping = None
        # Dummy parameter so that the model has parameters.
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def update_entities_to_extract(self, entities_to_extract):
        self.entities_to_extract = set(entities_to_extract)
        self.tokenizer.add_tokens(entities_to_extract)
        self.alliance_entities_loaded = False

    def set_tfidf_threshold(self, tfidf_threshold):
        self.tfidf_threshold = tfidf_threshold

    def set_min_matches(self, min_matches):
        self.min_matches = min_matches

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits = self.custom_entity_extraction(input_ids)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}

    def custom_entity_extraction(self, input_ids):
        """
        Produce logits at token level.
        The logits tensor should have shape (batch_size, seq_length, num_labels).
        """
        if self.load_entities_dynamically_fnc and not self.alliance_entities_loaded:
            entities_to_extract, name_to_curie_mapping = self.load_entities_dynamically_fnc()
            self.update_entities_to_extract(entities_to_extract)
            self.name_to_curie_mapping = name_to_curie_mapping
            self.alliance_entities_loaded = True
        batch_tokens = [self.tokenizer.convert_ids_to_tokens(seq) for seq in input_ids]
        logits_list = []

        global_token_counts = defaultdict(int)
        for tokens in batch_tokens:
            for token in tokens:
                if token in self.entities_to_extract:
                    global_token_counts[token] += 1

        for tokens in batch_tokens:
            # Initialize token-level logits: shape (num_tokens, num_labels).
            token_logits = torch.zeros(len(tokens), self.config.num_labels, device=input_ids.device)
            # Get the TF-IDF values for the document.
            document_text = " ".join(tokens)
            doc_tfidf = self.vectorizer.transform([document_text])
            # For each token in the document...
            for i, token in enumerate(tokens):
                if token in self.entities_to_extract:
                    # Use the in-document frequency (count) for this token.
                    token_count = global_token_counts[token]
                    # Get the tf-idf score for this token, if it exists in the fitted vocabulary.
                    if token in self.vectorizer.vocabulary_:
                        feature_index = self.vectorizer.vocabulary_[token]
                        tfidf_value = doc_tfidf[0, feature_index]
                    else:
                        tfidf_value = self.tfidf_threshold
                    # Check if the token meets both the frequency and TF-IDF threshold criteria.
                    if token_count >= self.min_matches and (
                            self.tfidf_threshold <= 0 or tfidf_value > self.tfidf_threshold):
                        token_logits[i, 1] = 1.0  # Label 1 for ENTITY detected.
            logits_list.append(token_logits)
            # Return a tensor of shape (batch_size, seq_length, num_labels).
        return torch.stack(logits_list, dim=0)
