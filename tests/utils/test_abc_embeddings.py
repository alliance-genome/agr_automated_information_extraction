from io import BytesIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils import abc_embeddings


def _parquet_bytes(rows):
    """Build a minimal ABC embedding parquet from ``rows`` of
    ``(embedding_list, is_document_level, content)``."""
    table = pa.table({
        "embedding": pa.array([r[0] for r in rows], type=pa.list_(pa.float32())),
        "is_document_level": pa.array([r[1] for r in rows], type=pa.bool_()),
        "content": pa.array([r[2] for r in rows], type=pa.string()),
    })
    buffer = BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def test_recipe_fields():
    # Only (profile, version) are stored on the model; everything else is a fixed
    # convention or read from the parquet.
    recipe = abc_embeddings.abc_embedding_recipe()
    assert recipe == {
        "embedding_profile": abc_embeddings.ABC_EMBEDDING_PROFILE,
        "embedding_version": abc_embeddings.ABC_EMBEDDING_VERSION,
    }


def test_is_abc_embedding_model():
    # ABC-embedding model: embedding_profile is set.
    assert abc_embeddings.is_abc_embedding_model({"embedding_profile": abc_embeddings.ABC_EMBEDDING_PROFILE})
    # Legacy / unavailable: profile null or absent.
    assert not abc_embeddings.is_abc_embedding_model({"embedding_profile": None})
    assert not abc_embeddings.is_abc_embedding_model({})
    assert not abc_embeddings.is_abc_embedding_model(None)


def test_pool_is_l2_normalized_chunk_mean_and_excludes_document_level():
    # Two paragraph rows + one document-level row (must be excluded from both the
    # pooled vector and the BoW text).
    parquet = _parquet_bytes([
        ([1.0, 0.0], False, "alpha beta"),
        ([0.0, 1.0], False, "gamma"),
        ([9.0, 9.0], True, "DOCLEVEL"),
    ])
    result = abc_embeddings.paragraph_pool_and_text(parquet)
    assert result is not None
    pooled, text = result
    # L2([1,0])=[1,0]; L2([0,1])=[0,1]; mean=[.5,.5]; L2(mean)=[0.7071,0.7071].
    np.testing.assert_allclose(pooled, np.array([0.70710678, 0.70710678], dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(pooled), 1.0, rtol=1e-5)
    assert text == "alpha beta gamma"
    assert pooled.dtype == np.float32


def test_pool_none_when_only_document_level():
    parquet = _parquet_bytes([([5.0, 6.0], True, "x")])
    assert abc_embeddings.paragraph_pool_and_text(parquet) is None


def test_pool_skips_null_embeddings_but_keeps_text():
    parquet = _parquet_bytes([
        (None, False, "no-vector text"),
        ([2.0, 0.0], False, "has vector"),
    ])
    pooled, text = abc_embeddings.paragraph_pool_and_text(parquet)
    # Only the one real vector contributes: L2([2,0]) = [1,0].
    np.testing.assert_allclose(pooled, np.array([1.0, 0.0], dtype=np.float32), rtol=1e-5)
    assert text == "no-vector text has vector"


def test_fulltext_profile_matches_legacy_constants():
    # The legacy module constants must keep pointing at the fulltext profile so
    # every existing call site and every live model resolves exactly as before.
    profile = abc_embeddings.FULLTEXT_PROFILE
    assert profile.name == abc_embeddings.ABC_EMBEDDING_PROFILE
    assert profile.version == abc_embeddings.ABC_EMBEDDING_VERSION
    assert profile.model_name == abc_embeddings.ABC_EMBEDDING_MODEL
    assert profile.dim == abc_embeddings.ABC_EMBEDDING_DIM
    # Fulltext embeddings are derived from the main converted Markdown.
    assert profile.required_source_file_class == "converted_merged_main"


def test_abstract_profile_has_no_source_file():
    # Abstract embeddings come from the reference record, not from a file, so
    # embedding_file.source_referencefile_id is NULL and there is nothing to match.
    profile = abc_embeddings.ABSTRACT_PROFILE
    assert profile.name == "classifier_abstract_title_abstract_single_chunk"
    assert profile.version == 1
    assert profile.model_name == "text-embedding-3-small"
    assert profile.dim == 1536
    assert profile.required_source_file_class is None


def test_get_profile_resolves_both_and_rejects_unknown():
    assert abc_embeddings.get_profile(
        abc_embeddings.ABC_EMBEDDING_PROFILE, 1) is abc_embeddings.FULLTEXT_PROFILE
    assert abc_embeddings.get_profile(
        "classifier_abstract_title_abstract_single_chunk", 1) is abc_embeddings.ABSTRACT_PROFILE
    # Unknown name, and known name with an unknown version, both resolve to None.
    assert abc_embeddings.get_profile("nope", 1) is None
    assert abc_embeddings.get_profile(abc_embeddings.ABC_EMBEDDING_PROFILE, 99) is None


def test_get_profile_by_name_resolves_the_registered_version():
    # The CLI takes a profile *name*; the version must come from the registry so a
    # third profile cannot be added without the trainer picking it up.
    assert abc_embeddings.get_profile_by_name(
        abc_embeddings.ABC_EMBEDDING_PROFILE) is abc_embeddings.FULLTEXT_PROFILE
    assert abc_embeddings.get_profile_by_name(
        abc_embeddings.ABSTRACT_PROFILE.name) is abc_embeddings.ABSTRACT_PROFILE
    assert abc_embeddings.get_profile_by_name("nope") is None


def test_recipe_can_stamp_the_abstract_profile():
    recipe = abc_embeddings.abc_embedding_recipe(
        profile_name=abc_embeddings.ABSTRACT_PROFILE.name,
        version=abc_embeddings.ABSTRACT_PROFILE.version)
    assert recipe == {
        "embedding_profile": "classifier_abstract_title_abstract_single_chunk",
        "embedding_version": 1,
    }


def test_abstract_chunk_text_is_the_single_convention():
    # This exact string is the coordination point with the ABC producer
    # (SCRUM-6140): BoW hashing is exact-token, so the format must match
    # production byte-for-byte. It lives here and nowhere else.
    assert abc_embeddings.abstract_chunk_text("A Title", "The abstract.") == "A Title\n\nThe abstract."
    # Missing pieces must not leave stray separators behind.
    assert abc_embeddings.abstract_chunk_text("", "Only abstract.") == "Only abstract."
    assert abc_embeddings.abstract_chunk_text("Only title", "") == "Only title"
    assert abc_embeddings.abstract_chunk_text("", "") == ""
    assert abc_embeddings.abstract_chunk_text("  T  ", "  A  ") == "T\n\nA"


def test_abstract_bow_profile_has_no_embedding_block():
    # SCRUM-5764: the evaluation showed embeddings add nothing for ZFIN molecular
    # probes, so the shipping profile is BoW over title+abstract with no dense
    # block at all -- and therefore no OpenAI dependency and no parquet.
    profile = abc_embeddings.ABSTRACT_BOW_PROFILE
    assert profile.name == "classifier_abstract_title_abstract_bow_only"
    assert profile.version == 1
    assert profile.use_embedding is False
    assert profile.dim == 0
    assert profile.model_name is None
    # Text comes from the reference record, not from an embedding parquet.
    assert profile.text_source == abc_embeddings.TEXT_SOURCE_REFERENCE_ABSTRACT
    assert profile.required_source_file_class is None


def test_embedding_profiles_read_their_text_from_the_parquet():
    for profile in (abc_embeddings.FULLTEXT_PROFILE, abc_embeddings.ABSTRACT_PROFILE):
        assert profile.use_embedding is True
        assert profile.text_source == abc_embeddings.TEXT_SOURCE_PARQUET
        assert profile.dim == 1536


def test_get_profile_resolves_the_bow_only_profile():
    assert abc_embeddings.get_profile(
        "classifier_abstract_title_abstract_bow_only", 1) is abc_embeddings.ABSTRACT_BOW_PROFILE
    assert abc_embeddings.get_profile_by_name(
        abc_embeddings.ABSTRACT_BOW_PROFILE.name) is abc_embeddings.ABSTRACT_BOW_PROFILE


def test_registered_profile_names_covers_every_profile():
    # The trainer's --embedding_profile choices come from here, so a newly
    # registered profile must not need a second edit to become selectable.
    names = abc_embeddings.registered_profile_names()
    assert names == sorted({abc_embeddings.FULLTEXT_PROFILE.name,
                            abc_embeddings.ABSTRACT_PROFILE.name,
                            abc_embeddings.ABSTRACT_BOW_PROFILE.name})


def test_bow_only_model_still_counts_as_a_profile_driven_model():
    # is_abc_embedding_model gates the whole non-BioWordVec path, so a BoW-only
    # abstract model must answer True even though it fetches no embedding.
    assert abc_embeddings.is_abc_embedding_model(
        {"embedding_profile": abc_embeddings.ABSTRACT_BOW_PROFILE.name})
