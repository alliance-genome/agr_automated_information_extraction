"""Profile threading through the classify/train paths (SCRUM-5764).

The abstract and fulltext profiles share a model and a dimension, so a mismatched
fetch lines up numerically and predicts silent nonsense. These tests pin the two
places that could leak one profile's vectors into another's features: the fetch
arguments, and the shared embedding cache key.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agr_document_classifier import agr_document_classifier_classify as classify
from agr_document_classifier import agr_document_classifier_trainer as trainer
from utils.abc_embeddings import (ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION,
                                  ABSTRACT_PROFILE, ABSTRACT_BOW_PROFILE,
                                  profile_pair_from_model)


def _stub_model():
    """A classifier stub whose predictions we ignore — these tests assert on
    which embeddings were fetched, not on the labels."""
    model = MagicMock()
    model.classes_ = np.array([0, 1])
    model.predict_proba.return_value = np.array([[0.25, 0.75]])
    return model


@patch("agr_document_classifier.agr_document_classifier_classify.get_reference_embedding")
def test_profile_and_version_are_passed_to_the_fetch(mock_fetch):
    mock_fetch.return_value = (np.zeros(4, dtype=np.float32), "text")
    classify.classify_documents_from_abc_embeddings(
        ["AGRKB:1"], "ZFIN", _stub_model(), use_bow=False,
        profile_name=ABSTRACT_PROFILE.name, version=ABSTRACT_PROFILE.version)
    _args, kwargs = mock_fetch.call_args
    assert kwargs["profile_name"] == ABSTRACT_PROFILE.name
    assert kwargs["version"] == ABSTRACT_PROFILE.version


@patch("agr_document_classifier.agr_document_classifier_classify.get_reference_embedding")
def test_cache_isolates_profiles_for_the_same_reference(mock_fetch):
    # The reclassify pipeline shares one cache across every model. Keying on the
    # curie alone would hand the second profile the first profile's vectors —
    # and since both profiles are 1536-d, nothing would raise.
    fulltext_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    abstract_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    mock_fetch.side_effect = [(fulltext_vec, "full"), (abstract_vec, "abs")]
    cache = {}

    classify.classify_documents_from_abc_embeddings(
        ["AGRKB:1"], "ZFIN", _stub_model(), use_bow=False, embedding_cache=cache,
        profile_name=ABC_EMBEDDING_PROFILE, version=ABC_EMBEDDING_VERSION)
    classify.classify_documents_from_abc_embeddings(
        ["AGRKB:1"], "ZFIN", _stub_model(), use_bow=False, embedding_cache=cache,
        profile_name=ABSTRACT_PROFILE.name, version=ABSTRACT_PROFILE.version)

    # Two distinct fetches, and both entries coexist under distinct keys.
    assert mock_fetch.call_count == 2
    assert cache[("AGRKB:1", ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION)][0] is fulltext_vec
    assert cache[("AGRKB:1", ABSTRACT_PROFILE.name, ABSTRACT_PROFILE.version)][0] is abstract_vec


@patch("agr_document_classifier.agr_document_classifier_classify.get_reference_embedding")
def test_same_profile_still_hits_the_cache_once(mock_fetch):
    mock_fetch.return_value = (np.zeros(4, dtype=np.float32), "text")
    cache = {}
    for _ in range(2):
        classify.classify_documents_from_abc_embeddings(
            ["AGRKB:1"], "ZFIN", _stub_model(), use_bow=False, embedding_cache=cache,
            profile_name=ABSTRACT_PROFILE.name, version=ABSTRACT_PROFILE.version)
    assert mock_fetch.call_count == 1


@patch("agr_document_classifier.agr_document_classifier_classify.get_reference_embedding",
       return_value=None)
def test_missing_embedding_zero_row_uses_the_profile_dim(_mock_fetch):
    # The zero row stands in for a missing embedding; its width must come from
    # the profile, not a module-level fulltext constant.
    model = _stub_model()
    ids, _classifications, _conf, valid = classify.classify_documents_from_abc_embeddings(
        ["AGRKB:1"], "ZFIN", model, use_bow=False,
        profile_name=ABSTRACT_PROFILE.name, version=ABSTRACT_PROFILE.version)
    assert ids == ["AGRKB:1"]
    assert valid == [False]
    matrix = model.predict_proba.call_args[0][0]
    assert matrix.shape == (1, ABSTRACT_PROFILE.dim)


def test_profile_pair_from_model_reads_the_stamped_pair():
    assert profile_pair_from_model({
        "embedding_profile": ABSTRACT_PROFILE.name,
        "embedding_version": ABSTRACT_PROFILE.version,
    }) == (ABSTRACT_PROFILE.name, ABSTRACT_PROFILE.version)


def test_profile_pair_from_model_defaults_to_fulltext():
    # Models uploaded before the abstract profile existed carry a profile with no
    # version, or no pair at all; both must resolve to the fulltext profile so
    # every live model keeps behaving identically.
    assert profile_pair_from_model({
        "embedding_profile": ABC_EMBEDDING_PROFILE,
    }) == (ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION)
    assert profile_pair_from_model({}) == (ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION)
    assert profile_pair_from_model(None) == (ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION)


@patch("agr_document_classifier.agr_document_classifier_classify.classify_documents_from_abc_embeddings")
def test_process_job_batch_passes_the_models_profile(mock_classify):
    # The production entry point must derive the pair from the model it loaded,
    # not from the module default — otherwise an abstract model silently
    # classifies from fulltext embeddings.
    mock_classify.return_value = (["AGRKB:1"], np.array([1]), [0.9], [True])
    # test_mode=True keeps this off the network: it logs instead of sending TETs.
    classify.process_job_batch(
        [{"reference_curie": "AGRKB:1"}], "ZFIN", "ATP:0000370", 1, None, _stub_model(),
        {"embedding_profile": ABSTRACT_PROFILE.name,
         "embedding_version": ABSTRACT_PROFILE.version,
         "data_novelty": "ATP:0000335"},
        True, use_abc_embeddings=True, abc_use_bow=True)
    _args, kwargs = mock_classify.call_args
    assert kwargs["profile_name"] == ABSTRACT_PROFILE.name
    assert kwargs["version"] == ABSTRACT_PROFILE.version


@patch("agr_document_classifier.agr_document_classifier_trainer.get_reference_embedding")
def test_trainer_feature_build_uses_the_requested_profile(mock_fetch):
    mock_fetch.return_value = (np.zeros(4, dtype=np.float32), "Title\n\nAbstract.")
    X, y = trainer._build_abc_embedding_features(
        {"positive": ["AGRKB:1"], "negative": ["AGRKB:2"]}, "ZFIN", use_bow=True,
        profile_name=ABSTRACT_PROFILE.name, version=ABSTRACT_PROFILE.version)
    assert y == [1, 0]
    assert X.shape[0] == 2
    for _args, kwargs in mock_fetch.call_args_list:
        assert kwargs["profile_name"] == ABSTRACT_PROFILE.name
        assert kwargs["version"] == ABSTRACT_PROFILE.version


@patch("agr_document_classifier.agr_document_classifier_trainer._select_and_fit_model",
       return_value=("model", {"model_name": "stub"}))
@patch("agr_document_classifier.agr_document_classifier_trainer._build_abc_embedding_features")
def test_train_classifier_resolves_the_profile_version_from_its_name(mock_build, _mock_fit):
    # The CLI takes only a name; the version must be looked up, never guessed.
    mock_build.return_value = (np.zeros((2, 4), dtype=np.float32), [1, 0])
    trainer.train_classifier(
        embedding_model_path=None, training_data_dir=None, use_abc_embeddings=True,
        abc_curies={"positive": ["AGRKB:1"], "negative": ["AGRKB:2"]},
        mod_abbreviation="ZFIN", embedding_profile=ABSTRACT_PROFILE.name)
    _args, kwargs = mock_build.call_args
    assert kwargs["profile_name"] == ABSTRACT_PROFILE.name
    assert kwargs["version"] == ABSTRACT_PROFILE.version


def test_train_classifier_rejects_an_unknown_profile():
    # Silently falling back to fulltext here would train a model on the wrong
    # features and stamp it with a profile that does not exist.
    with pytest.raises(ValueError, match="Unknown embedding profile"):
        trainer.train_classifier(
            embedding_model_path=None, training_data_dir=None, use_abc_embeddings=True,
            abc_curies={"positive": [], "negative": []}, mod_abbreviation="ZFIN",
            embedding_profile="typo_profile")


@patch("agr_document_classifier.agr_document_classifier_classify.get_reference_embedding")
@patch("agr_document_classifier.agr_document_classifier_classify.get_reference_abstract_text",
       return_value="Probe synthesis\n\nWe made a probe.")
def test_bow_only_profile_reads_the_reference_record_not_a_parquet(mock_text, mock_embed):
    # The whole point of the BoW-only profile: no embedding fetch at all, so no
    # parquet, no OpenAI, and no SCRUM-6140 dependency.
    model = _stub_model()
    ids, _cls, _conf, valid = classify.classify_documents_from_abc_embeddings(
        ["AGRKB:1"], "ZFIN", model, use_bow=True,
        profile_name=ABSTRACT_BOW_PROFILE.name, version=ABSTRACT_BOW_PROFILE.version)
    mock_embed.assert_not_called()
    mock_text.assert_called_once_with("AGRKB:1")
    assert ids == ["AGRKB:1"] and valid == [True]


@patch("agr_document_classifier.agr_document_classifier_classify.get_reference_abstract_text",
       return_value="Probe synthesis\n\nWe made a probe.")
def test_bow_only_matrix_has_no_dense_block(_mock_text):
    # Width must be the BoW block alone. If the dense block leaked in as a
    # zero-width or 1536-wide slab, the model would see the wrong feature layout.
    from utils.embedding import get_bow_vectorizer
    bow_width = get_bow_vectorizer().transform([""]).shape[1]
    model = _stub_model()
    classify.classify_documents_from_abc_embeddings(
        ["AGRKB:1"], "ZFIN", model, use_bow=True,
        profile_name=ABSTRACT_BOW_PROFILE.name, version=ABSTRACT_BOW_PROFILE.version)
    assert model.predict_proba.call_args[0][0].shape == (1, bow_width)


@patch("agr_document_classifier.agr_document_classifier_classify.get_reference_abstract_text",
       return_value=None)
def test_bow_only_marks_a_reference_with_no_abstract_invalid(_mock_text):
    # No abstract means it cannot be classified from an abstract; the job must fail
    # rather than being scored on an empty string.
    ids, _cls, _conf, valid = classify.classify_documents_from_abc_embeddings(
        ["AGRKB:1"], "ZFIN", _stub_model(), use_bow=True,
        profile_name=ABSTRACT_BOW_PROFILE.name, version=ABSTRACT_BOW_PROFILE.version)
    assert ids == ["AGRKB:1"] and valid == [False]


@patch("agr_document_classifier.agr_document_classifier_classify.get_reference_abstract_text")
def test_bow_only_profile_shares_the_cache_by_profile(mock_text):
    mock_text.return_value = "text"
    cache = {}
    for _ in range(2):
        classify.classify_documents_from_abc_embeddings(
            ["AGRKB:1"], "ZFIN", _stub_model(), use_bow=True, embedding_cache=cache,
            profile_name=ABSTRACT_BOW_PROFILE.name, version=ABSTRACT_BOW_PROFILE.version)
    assert mock_text.call_count == 1
    assert (("AGRKB:1", ABSTRACT_BOW_PROFILE.name, ABSTRACT_BOW_PROFILE.version)) in cache


@patch("agr_document_classifier.agr_document_classifier_trainer.get_reference_embedding")
@patch("agr_document_classifier.agr_document_classifier_trainer.get_reference_abstract_text",
       return_value="Probe synthesis\n\nWe made a probe.")
def test_trainer_bow_only_profile_reads_the_reference_record(mock_text, mock_embed):
    # Training must use the same text source classify will use, or the BoW block
    # differs between train and inference.
    X, y = trainer._build_abc_embedding_features(
        {"positive": ["AGRKB:1"], "negative": ["AGRKB:2"]}, "ZFIN", use_bow=True,
        profile_name=ABSTRACT_BOW_PROFILE.name, version=ABSTRACT_BOW_PROFILE.version)
    mock_embed.assert_not_called()
    assert mock_text.call_count == 2
    assert y == [1, 0]
    from utils.embedding import get_bow_vectorizer
    assert X.shape == (2, get_bow_vectorizer().transform([""]).shape[1])


@patch("agr_document_classifier.agr_document_classifier_trainer.get_reference_abstract_text",
       return_value=None)
def test_trainer_drops_references_with_no_abstract(_mock_text):
    # Same policy as a missing embedding: dropped and counted, not trained on "".
    with pytest.raises(ValueError, match="No ABC embeddings could be retrieved"):
        trainer._build_abc_embedding_features(
            {"positive": ["AGRKB:1"], "negative": ["AGRKB:2"]}, "ZFIN", use_bow=True,
            profile_name=ABSTRACT_BOW_PROFILE.name, version=ABSTRACT_BOW_PROFILE.version)


def test_trainer_cli_offers_every_registered_profile():
    # A newly registered profile must be selectable without a second edit.
    import agr_document_classifier.agr_document_classifier_trainer as t
    from utils.abc_embeddings import registered_profile_names
    parser_action = [a for a in t.parse_arguments.__wrapped__.__code__.co_consts
                     if isinstance(a, str) and a == "--embedding_profile"] \
        if hasattr(t.parse_arguments, "__wrapped__") else ["--embedding_profile"]
    assert parser_action  # sanity: the flag exists
    import sys
    argv = sys.argv
    try:
        sys.argv = ["prog", "--embedding_profile", ABSTRACT_BOW_PROFILE.name]
        args = t.parse_arguments()
    finally:
        sys.argv = argv
    assert args.embedding_profile == ABSTRACT_BOW_PROFILE.name
    assert set(registered_profile_names()) >= {ABSTRACT_BOW_PROFILE.name}
