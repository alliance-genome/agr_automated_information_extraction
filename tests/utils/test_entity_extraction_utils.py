"""Tests for the allele topic helpers in utils.entity_extraction_utils.

WB allele extraction involves two ATP topics: ABC keys the extraction jobs on
the classical allele topic (ATP:0000285), while the extraction and TF-IDF
models are registered under the generic allele topic (ATP:0000006).
normalize_allele_topic() maps any allele topic to the model topic so jobs are
neither dropped by topic filters nor looked up under a topic with no model
(SCRUM-6596). Reporting to ABC always uses ABC_ALLELE_TOPIC (ATP:0000285).
"""

from utils.entity_extraction_utils import (
    ABC_ALLELE_TOPIC,
    MODEL_ALLELE_TOPIC,
    is_allele_topic,
    normalize_allele_topic,
)


def test_is_allele_topic_accepts_generic_and_classical():
    assert is_allele_topic("ATP:0000006")
    assert is_allele_topic("ATP:0000285")


def test_is_allele_topic_rejects_other_topics():
    assert not is_allele_topic("ATP:0000005")
    assert not is_allele_topic(None)


def test_normalize_allele_topic_maps_allele_topics_to_model_topic():
    assert MODEL_ALLELE_TOPIC == "ATP:0000006"
    assert normalize_allele_topic("ATP:0000285") == MODEL_ALLELE_TOPIC
    assert normalize_allele_topic("ATP:0000006") == MODEL_ALLELE_TOPIC


def test_normalize_allele_topic_leaves_non_allele_topics_unchanged():
    for topic in ("ATP:0000005", "ATP:0000027", "ATP:0000110", None):
        assert normalize_allele_topic(topic) == topic


def test_reporting_topic_is_classical_allele():
    assert ABC_ALLELE_TOPIC == "ATP:0000285"
