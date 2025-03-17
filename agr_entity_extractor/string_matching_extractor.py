from collections import defaultdict

import dill
import torch
from transformers import PreTrainedModel, PretrainedConfig

from utils.abc_utils import upload_ml_model, download_abc_model


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
                 tokenizer, vectorizer, entities_to_extract, match_uppercase: bool = False):
        super().__init__(config)
        self.config = config
        self.tfidf_threshold = tfidf_threshold
        self.match_uppercase = match_uppercase
        self.min_matches = min_matches
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.entities_to_extract = set(entities_to_extract) if entities_to_extract else None
        # Dummy parameter so that the model has parameters.
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def set_entities_to_extract(self, entities_to_extract):
        self.entities_to_extract = set(entities_to_extract)
        self.tokenizer.tokenizer.add_tokens(entities_to_extract)
        self.vectorizer.tokenizer.tokenizer.add_tokens(entities_to_extract)

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
            doc_tfidf = self.vectorizer.transform([tokens])
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Upload the entity extractor model to the Alliance ML API")
    parser.add_argument("-m", "--mod-abbreviation", required=True,
                        help="The MOD abbreviation (e.g., FB, WB, SGD, etc.)")
    parser.add_argument("--min-matches", required=True, help="Minimum number of matches required for an "
                                                             "entity to be extracted")
    parser.add_argument("--tfidf-threshold", required=True, help="TF-IDF threshold for entity extraction")
    parser.add_argument("-t", "--topic", required=True, help="The topic of the model")
    args = parser.parse_args()

    tfidf_vectorizer_model_file_path = (f"/data/agr_entity_extraction/tfidf_vectorization_"
                                        f"{args.mod_abbreviation}_notopic.dpkl")
    download_abc_model(mod_abbreviation=args.mod_abbreviation, topic=None,
                       output_path=tfidf_vectorizer_model_file_path, task_type="tfidf_vectorization")

    tfidf_vectorizer = dill.load(open(tfidf_vectorizer_model_file_path, "rb"))

    entity_extraction_model_file_path = (f"/data/agr_entity_extraction/biocuration_entity_extraction_"
                                         f"{args.mod_abbreviation}_{args.topic.replace(':', '_')}.dpkl")

    # Initialize the model
    config = AllianceStringMatchingEntityExtractorConfig()
    model = AllianceStringMatchingEntityExtractor(
        config=config,
        min_matches=args.min_matches,
        tfidf_threshold=args.tfidf_threshold,
        tokenizer=tfidf_vectorizer.tokenizer,
        vectorizer=tfidf_vectorizer,
        entities_to_extract=None
    )

    # Serialize the model
    with open(entity_extraction_model_file_path, "wb") as file:
        dill.dump(model, file)

    stats = {
        "model_name": "Alliance String Matching Entity Extractor",
        "average_precision": None,
        "average_recall": None,
        "average_f1": None,
        "best_params": None,
    }
    upload_ml_model(task_type="biocuration_entity_extraction", mod_abbreviation=args.mod_abbreviation,
                    model_path=entity_extraction_model_file_path, stats=stats, topic=args.topic, file_extension="dpkl")


if __name__ == "__main__":
    main()
