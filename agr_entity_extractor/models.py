from collections import defaultdict

import torch
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer


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


DEFAULT_MAX_LENGTH = 512  # Fallback maximum length if none is provided


class CustomTokenizer:
    def __init__(self, model_name, additional_tokens=None, model_max_length=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if additional_tokens:
            self.tokenizer.add_tokens(additional_tokens)
        # Allow an override; otherwise, prefer the tokenizer’s model_max_length or a default value
        self.model_max_length = DEFAULT_MAX_LENGTH
        self.is_fast = self.tokenizer.is_fast

    def __call__(self, text, *args, **kwargs):
        # If the input is already a list, process each element.
        if not kwargs:
            tokens = []
            for start in range(0, len(text), self.model_max_length):
                chunk = text[start: start + self.model_max_length]
                chunk_tokens = self.tokenizer.tokenize(chunk)
                tokens.extend(chunk_tokens)
            return tokens
        if isinstance(text, list):
            return self._batch_encode(text, *args, **kwargs)
        else:
            return self._encode(text, *args, **kwargs)

    def _batch_encode(self, texts, *args, **kwargs):
        # Process each text individually and then aggregate the outputs.
        batch = [self._encode(t, *args, **kwargs) for t in texts]
        result = {}
        # (Assumes that each output dict has the same keys.)
        for key in batch[0]:
            result[key] = [item[key] for item in batch]
        return result

    def _encode(self, text, *args, **kwargs):
        # Extract some common kwargs; note that we "pop" them so they do not remain in kwargs.
        padding = kwargs.pop("padding", True)
        truncation = kwargs.pop("truncation", True)
        return_tensors = kwargs.pop("return_tensors", None)
        return_special_tokens_mask = kwargs.pop("return_special_tokens_mask", False)

        tokens = []
        overflow_mapping = []

        # Process long text: split the input into chunks (here we use a simple character‐based split)
        for start in range(0, len(text), self.model_max_length):
            chunk = text[start: start + self.model_max_length]
            chunk_tokens = self.tokenizer.tokenize(chunk)
            tokens.extend(chunk_tokens)
            overflow_mapping.extend(
                [start] * len(chunk_tokens))  # map tokens back to the start char index of their chunk

        # Convert tokens into numerical IDs.
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Apply truncation if the token list is too long.
        if truncation and len(input_ids) > self.model_max_length:
            input_ids = input_ids[:self.model_max_length]
            overflow_mapping = overflow_mapping[:self.model_max_length]

        # Apply padding if needed.
        if padding and len(input_ids) < self.model_max_length:
            padding_length = self.model_max_length - len(input_ids)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids.extend([pad_id] * padding_length)
            overflow_mapping.extend([-1] * padding_length)

        # Build attention mask (1 for tokens, 0 for padding)
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]

        # Create token type IDs (all zeros, assuming a single sequence)
        token_type_ids = [0] * len(input_ids)

        # Optionally compute the special tokens mask.
        special_tokens_mask = None
        if return_special_tokens_mask:
            # If available, use the tokenizer’s cls and sep tokens; also treat pad (or 0) as special.
            cls_id = getattr(self.tokenizer, "cls_token_id", None)
            sep_id = getattr(self.tokenizer, "sep_token_id", None)
            special_ids = {cls_id, sep_id, self.tokenizer.pad_token_id, 0}
            special_tokens_mask = [1 if token_id in special_ids else 0 for token_id in input_ids]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "overflow_to_sample_mapping": overflow_mapping,
        }
        if special_tokens_mask is not None:
            result["special_tokens_mask"] = special_tokens_mask

        # Convert lists into tensors if requested.
        if return_tensors == "pt":
            import torch
            converted = {}
            # For standard keys we convert to a tensor with a batch dimension.
            for key in result.keys():
                converted[key] = torch.tensor([result[key]])
            # Other keys (like overflow mapping) are left as lists.
            # You can extend this conversion as needed.
            return converted
        elif return_tensors == "tf":
            import tensorflow as tf
            converted = {}
            for key in result.keys():
                converted[key] = tf.constant([result[key]])
            return converted

        return result

    def convert_ids_to_tokens(self, seq):
        return self.tokenizer.convert_ids_to_tokens(seq)

    def add_tokens(self, tokens):
        self.tokenizer.add_tokens(tokens)
        self.model_max_length = getattr(self.tokenizer, "model_max_length", self.model_max_length)

    def __getattr__(self, name):
        # Delegate attribute access to the underlying tokenizer so that methods
        # expected by the Hugging Face pipeline are available.
        return getattr(self.tokenizer, name)


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
                 tokenizer, vectorizer, entities_to_extract, load_entities_dynamically_fnc = None,
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
        self.vectorizer.tokenizer.add_tokens(entities_to_extract)

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
            doc_tfidf = self.vectorizer.transform(tokens)
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
