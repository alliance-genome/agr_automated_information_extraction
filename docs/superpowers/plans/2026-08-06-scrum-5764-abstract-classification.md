# SCRUM-5764 — ZFIN Molecular Probe Abstract-Only Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the document classifier able to train and classify from **abstract-only** ABC embeddings (a second embedding profile), then measure three approaches on ZFIN's 950 labelled molecular-probe references to decide what to ship.

**Architecture:** The classifier already consumes ABC precomputed embeddings via a `(profile, version)` pair stored on `ml_model` (SCRUM-5781). That pair is currently stored but never *used* — every fetch falls back to the hardcoded fulltext profile. This plan turns the profile into real data: a registry in `utils/abc_embeddings.py`, threaded from model metadata through `get_reference_embedding`, with a per-profile source-file constraint (abstract embeddings have no source referencefile). Feature shape is unchanged — L2-normalized chunk-mean pool concatenated with the stateless hashed BoW block, where BoW hashes the parquet's own `content` column, so pointing `content` at title+abstract makes BoW abstract-aware with zero BoW code changes.

**Tech Stack:** Python 3.12, scikit-learn / XGBoost / LightGBM, `pyarrow` 17.0.0 (parquet), `openai` (new — embeddings + LLM arm), ABC REST API, pytest.

## Global Constraints

- **Never commit without explicit user permission** (repo CLAUDE.md). The `Commit` steps below are pre-authored for when permission is given — do not run them otherwise.
- **The supervised model is always embedding + BoW.** There is no embeddings-only candidate. `_build_abc_embedding_features` already forces `use_bow=True` regardless of the flag (`agr_document_classifier_trainer.py:145-149`); keep that invariant for every profile.
- **Every existing production model must behave byte-for-byte identically.** The five live FB models (ml_model_id 56-60) carry `embedding_profile = classifier_fulltext_paragraph_chunk_refs_excluded_md_cleaned`, `embedding_version = 1`. Legacy BioWordVec models have `embedding_profile` NULL. Both paths stay unchanged.
- **Profile names are safety-critical.** Fulltext and abstract profiles are both `text-embedding-3-small`, 1536-d — dims match, so a mismatched fetch produces *silently wrong predictions with no error*. Never default a profile lookup to "whatever is available".
- PEP8, 4-space indent, no trailing whitespace, lines under 120 chars. `flake8` config: `setup.cfg` (`max-complexity = 18`).
- Verify with `python3 -m flake8 .` and `pytest tests/` before each commit.
- Absolute values fixed by this plan: abstract profile name `classifier_abstract_title_abstract_single_chunk`, version `1`, model `text-embedding-3-small`, dim `1536`, BoW `n_features = 2**18` (`utils/embedding.py:17-18`).

## External dependencies (block Phase 3b and shipping, not Phases 1/2/4)

1. **ATP term for "molecular probe" does not exist.** Ticket SCRUM-5764: "Still need to make new term for topic". `--datatype_train` and `create_dataset`/`add_entry_to_dataset` are keyed by topic ATP ID, so no ABC dataset upload and no model registration is possible until Ceri Van Slyke gets the term minted. **Offline evaluation does not need it.**
2. **`embedding_file` catalog rows cannot be created over HTTP.** `embedding_file_router.py` is read-only by design: *"Creation is intentionally NOT exposed over HTTP: embeddings are generated and registered ABC-internally only (`embedding_file_crud.create_or_update`)"*. `POST /reference/referencefile/file_upload/` uploads the parquet, but without a catalog row `show_all` returns no `profile_name`, so `get_reference_embedding` skips it. Registration needs Shuai Weng (SCRUM-6140) or a new creation route.
3. **Two conventions must be agreed with Shuai before any parquet is generated for real:** the profile name, and the exact `content` string format. BoW hashing is exact-token, so a different separator or casing in production yields a different BoW block and silently degrades a model trained on ours. Task 7 puts that convention in exactly one function so it changes in one place.

## File Structure

| File | Responsibility |
|---|---|
| `utils/abc_embeddings.py` (modify) | Profile registry + per-profile metadata + the abstract `content` convention. Single source of truth. |
| `utils/abc_utils.py` (modify, `get_reference_embedding` at 986-1038) | Resolve the profile and apply its source constraint when selecting the embedding referencefile. |
| `agr_document_classifier/agr_document_classifier_classify.py` (modify) | Pass the model's profile into the fetch; profile-aware cache key and zero-row width. |
| `agr_document_classifier/agr_document_classifier_reclassify.py` (modify, 159-160, 197-198) | Pass profile through at both call sites. |
| `agr_document_classifier/agr_document_classifier_trainer.py` (modify) | `--embedding_profile` CLI; thread to feature build; stamp the recipe at upload. |
| `scripts/prepare_zfin_probe_training_set.py` (create) | Normalize Ceri's CSV, resolve xrefs, fetch title+abstract, report coverage. |
| `scripts/generate_abstract_embeddings.py` (create) | Embed title+abstract, write ABC-format parquet, optionally upload. |
| `scripts/evaluate_probe_classifiers.py` (create) | Three-arm evaluation on one fixed stratified split + threshold report. |
| `tests/utils/test_abc_embeddings.py` (modify) | Registry + recipe + content-convention tests. |
| `tests/utils/test_get_reference_embedding.py` (modify) | Per-profile source-constraint selection tests. |
| `tests/agr_document_classifier/test_abstract_profile_plumbing.py` (create) | Profile threading + cache-key isolation tests. |

---

## Phase 1 — Profile-aware embedding plumbing

*No external blockers. Required regardless of which evaluation arm wins.*

### Task 1: Embedding profile registry

**Files:**
- Modify: `utils/abc_embeddings.py:35-79`
- Test: `tests/utils/test_abc_embeddings.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `EmbeddingProfile` frozen dataclass with fields `name: str`, `version: int`, `model_name: str`, `dim: int`, `required_source_file_class: Optional[str]`. Module constants `FULLTEXT_PROFILE`, `ABSTRACT_PROFILE`. `get_profile(profile_name: str, version: int) -> Optional[EmbeddingProfile]`. `abstract_chunk_text(title: str, abstract: str) -> str`. `abc_embedding_recipe(profile_name: str = ABC_EMBEDDING_PROFILE, version: int = ABC_EMBEDDING_VERSION) -> dict`. Existing names `ABC_EMBEDDING_PROFILE`, `ABC_EMBEDDING_VERSION`, `ABC_EMBEDDING_MODEL`, `ABC_EMBEDDING_DIM`, `MAIN_SOURCE_FILE_CLASS` all keep their current values.

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_abc_embeddings.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/utils/test_abc_embeddings.py -v`
Expected: FAIL — `AttributeError: module 'utils.abc_embeddings' has no attribute 'FULLTEXT_PROFILE'`. The four pre-existing tests in the file must still PASS.

- [ ] **Step 3: Implement the registry**

In `utils/abc_embeddings.py`, add `from dataclasses import dataclass` to the imports, then **replace** the constants block at lines 35-49 with:

```python
@dataclass(frozen=True)
class EmbeddingProfile:
    """One ABC embedding profile: what text was embedded, with which model, and
    how to recognise its parquet among a reference's referencefiles.

    ``required_source_file_class`` is the ``file_class`` the embedding's ``source``
    referencefile must have. It is ``None`` for profiles with no source file at
    all — an abstract embedding is derived from the reference record, so
    ``embedding_file.source_referencefile_id`` is NULL (the column is nullable
    precisely to allow this) and there is nothing to match against.
    """

    name: str
    version: int
    model_name: str
    dim: int
    required_source_file_class: Optional[str]


# --- The fulltext profile (SCRUM-6142): one embedding per paragraph chunk of the
# main PDF's converted Markdown. This is what every model in production today
# was trained on. ---
FULLTEXT_PROFILE = EmbeddingProfile(
    name="classifier_fulltext_paragraph_chunk_refs_excluded_md_cleaned",
    version=1,
    model_name="text-embedding-3-small",
    dim=1536,
    required_source_file_class="converted_merged_main",
)

# --- The abstract profile (SCRUM-5764): a single chunk holding title+abstract,
# for classification that must happen before any PDF exists. Same model and dim
# as the fulltext profile, which is exactly why the profile name matters: a
# mismatched fetch would line up dimensionally and predict silent nonsense. ---
ABSTRACT_PROFILE = EmbeddingProfile(
    name="classifier_abstract_title_abstract_single_chunk",
    version=1,
    model_name="text-embedding-3-small",
    dim=1536,
    required_source_file_class=None,
)

_PROFILES = {(p.name, p.version): p for p in (FULLTEXT_PROFILE, ABSTRACT_PROFILE)}

# Legacy aliases. Every existing call site and every model with
# embedding_profile NULL resolves to the fulltext profile, unchanged.
ABC_EMBEDDING_PROFILE = FULLTEXT_PROFILE.name
ABC_EMBEDDING_VERSION = FULLTEXT_PROFILE.version
ABC_EMBEDDING_MODEL = FULLTEXT_PROFILE.model_name
ABC_EMBEDDING_DIM = FULLTEXT_PROFILE.dim
# How the dense per-reference vector is built: L2-normalize each chunk embedding,
# average them, and L2-normalize the mean (SCRUM-6052 recipe). The document-level
# parquet row is ignored. Applies to every profile.
ABC_EMBEDDING_POOLING = "l2_chunk_mean"
MAIN_SOURCE_FILE_CLASS = FULLTEXT_PROFILE.required_source_file_class


def get_profile(profile_name: str, version: int) -> Optional[EmbeddingProfile]:
    """Return the registered profile for ``(profile_name, version)``, or ``None``
    when the pair is unknown. Callers must treat ``None`` as "refuse to guess":
    profiles can share dimensions, so falling back to an arbitrary profile would
    produce silently wrong features rather than an error."""
    return _PROFILES.get((profile_name, version))


def abstract_chunk_text(title: str, abstract: str) -> str:
    """Return the exact chunk text embedded (and hashed for BoW) by the abstract
    profile: title and abstract joined by a blank line, each stripped, with empty
    parts omitted so no stray separator survives.

    This is the coordination point with the ABC producer (SCRUM-6140). The hashed
    BoW block is built from this same string, and hashing is exact-token, so the
    format must match production byte-for-byte or a model trained here will get a
    different BoW block at classify time. Keep it defined only in this function.
    """
    parts = [part.strip() for part in (title or "", abstract or "")]
    return "\n\n".join(part for part in parts if part)
```

Then change the `abc_embedding_recipe` signature (currently line 70) to accept the pair:

```python
def abc_embedding_recipe(profile_name: str = ABC_EMBEDDING_PROFILE,
                         version: int = ABC_EMBEDDING_VERSION) -> dict:
```

and its body to `return {"embedding_profile": profile_name, "embedding_version": version}`. Leave the docstring's explanation intact.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/utils/test_abc_embeddings.py -v`
Expected: PASS, all tests including the four pre-existing ones (`test_recipe_fields` still passes because the defaults are unchanged).

- [ ] **Step 5: Verify nothing else regressed, then commit**

Run: `pytest tests/ -q && python3 -m flake8 .`
Expected: full suite passes, no lint findings.

```bash
git add utils/abc_embeddings.py tests/utils/test_abc_embeddings.py
git commit -m "feat(SCRUM-5764): add embedding profile registry with abstract profile"
```

---

### Task 2: Per-profile source constraint in `get_reference_embedding`

**Files:**
- Modify: `utils/abc_utils.py:986-1028`
- Test: `tests/utils/test_get_reference_embedding.py`

**Interfaces:**
- Consumes: `get_profile`, `ABSTRACT_PROFILE`, `MAIN_SOURCE_FILE_CLASS` from Task 1.
- Produces: `get_reference_embedding(reference_curie, mod_abbreviation, profile_name=ABC_EMBEDDING_PROFILE, version=ABC_EMBEDDING_VERSION) -> Optional[Tuple[np.ndarray, str]]` — signature unchanged, behaviour now profile-dependent.

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_get_reference_embedding.py` (the `_embedding_row` helper already
supports a `None` source):

```python
from utils.abc_embeddings import ABSTRACT_PROFILE


@patch("utils.abc_utils.paragraph_pool_and_text",
       return_value=(np.array([0.0, 1.0], dtype=np.float32), "Title\n\nAbstract."))
@patch("utils.abc_utils.get_file_from_abc_reffile_obj", return_value=b"parquet-bytes")
@patch("utils.abc_utils._show_all_for_reference")
def test_abstract_profile_accepts_row_with_no_source(mock_show_all, mock_download, _mock_pool):
    # An abstract embedding has no source referencefile, so requiring
    # converted_merged_main would reject it outright.
    mock_show_all.return_value = [
        _embedding_row(7, ABSTRACT_PROFILE.name, ABSTRACT_PROFILE.version, None),
    ]
    result = abc_utils.get_reference_embedding(
        "AGRKB:1", "ZFIN",
        profile_name=ABSTRACT_PROFILE.name, version=ABSTRACT_PROFILE.version)
    assert result is not None
    pooled, text = result
    np.testing.assert_allclose(pooled, np.array([0.0, 1.0], dtype=np.float32))
    assert text == "Title\n\nAbstract."
    assert mock_download.call_args[0][0]["referencefile_id"] == 7


@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
@patch("utils.abc_utils._show_all_for_reference")
def test_abstract_profile_ignores_fulltext_rows(mock_show_all, mock_download):
    # Both profiles are 1536-d, so picking the wrong one would NOT raise — it
    # would silently predict from the wrong features. The profile name is the
    # only thing preventing that.
    mock_show_all.return_value = [
        _embedding_row(1, ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION, "converted_merged_main"),
    ]
    assert abc_utils.get_reference_embedding(
        "AGRKB:1", "ZFIN",
        profile_name=ABSTRACT_PROFILE.name, version=ABSTRACT_PROFILE.version) is None
    mock_download.assert_not_called()


@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
@patch("utils.abc_utils._show_all_for_reference")
def test_fulltext_profile_still_rejects_sourceless_rows(mock_show_all, mock_download):
    # The fulltext constraint must not be loosened by making it per-profile.
    mock_show_all.return_value = [
        _embedding_row(1, ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION, None),
    ]
    assert abc_utils.get_reference_embedding("AGRKB:1", "FB") is None
    mock_download.assert_not_called()


@patch("utils.abc_utils.get_file_from_abc_reffile_obj")
@patch("utils.abc_utils._show_all_for_reference")
def test_unknown_profile_requires_main_source(mock_show_all, mock_download):
    # An unregistered profile must not be treated as "no constraint" — that would
    # make a typo in a model's embedding_profile silently match anything.
    mock_show_all.return_value = [
        _embedding_row(1, "unregistered_profile", 1, None),
    ]
    assert abc_utils.get_reference_embedding(
        "AGRKB:1", "FB", profile_name="unregistered_profile", version=1) is None
    mock_download.assert_not_called()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/utils/test_get_reference_embedding.py -v`
Expected: `test_abstract_profile_accepts_row_with_no_source` FAILS (returns `None` — the hardcoded `converted_merged_main` check rejects the sourceless row). The other three may already pass; they are regression guards.

- [ ] **Step 3: Make the source constraint profile-driven**

In `utils/abc_utils.py`, extend the import at line 16 to include `get_profile`. Then replace the selection loop body at lines 1013-1025 with:

```python
    # Which source file_class this profile's parquet must be derived from. An
    # unregistered profile falls back to the strict fulltext constraint rather
    # than "no constraint", so a typo in a model's embedding_profile cannot
    # silently match an unrelated embedding row.
    profile = get_profile(profile_name, version)
    required_source_file_class = (profile.required_source_file_class if profile
                                  else MAIN_SOURCE_FILE_CLASS)

    embedding_ref_file = None
    for ref_file in resp_obj:
        if ref_file.get("file_class") != "embedding":
            continue
        if ref_file.get("profile_name") != profile_name:
            continue
        if ref_file.get("version") != version:
            continue
        if required_source_file_class is not None:
            source = ref_file.get("source") or {}
            if source.get("file_class") != required_source_file_class:
                continue
        embedding_ref_file = ref_file
        break
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/utils/test_get_reference_embedding.py -v`
Expected: PASS, all tests including the five pre-existing ones.

- [ ] **Step 5: Commit**

Run: `pytest tests/ -q && python3 -m flake8 .`

```bash
git add utils/abc_utils.py tests/utils/test_get_reference_embedding.py
git commit -m "feat(SCRUM-5764): make embedding source constraint per-profile"
```

---

### Task 3: Profile-aware cache key and zero-row width in classify

**Files:**
- Modify: `agr_document_classifier/agr_document_classifier_classify.py:107-156`
- Test: `tests/agr_document_classifier/test_abstract_profile_plumbing.py` (create)

**Interfaces:**
- Consumes: `get_profile`, `ABSTRACT_PROFILE` (Task 1); profile-aware `get_reference_embedding` (Task 2).
- Produces: `classify_documents_from_abc_embeddings(reference_curies, mod_abbr, classifier_model, use_bow=False, embedding_cache=None, profile_name=ABC_EMBEDDING_PROFILE, version=ABC_EMBEDDING_VERSION)`. `embedding_cache` keys become the tuple `(reference_curie, profile_name, version)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/agr_document_classifier/test_abstract_profile_plumbing.py`:

```python
from unittest.mock import MagicMock, patch

import numpy as np

from agr_document_classifier import agr_document_classifier_classify as classify
from utils.abc_embeddings import (ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION,
                                 ABSTRACT_PROFILE)


def _stub_model():
    """A classifier stub whose predictions we ignore — these tests assert on
    which embeddings were fetched, not on the labels."""
    model = MagicMock()
    model.predict.return_value = np.array([1])
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
    ids, _classifications, _conf, valid = classify.classify_documents_from_abc_embeddings(
        ["AGRKB:1"], "ZFIN", _stub_model(), use_bow=False,
        profile_name=ABSTRACT_PROFILE.name, version=ABSTRACT_PROFILE.version)
    assert ids == ["AGRKB:1"]
    assert valid == [False]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/agr_document_classifier/test_abstract_profile_plumbing.py -v`
Expected: FAIL — `TypeError: classify_documents_from_abc_embeddings() got an unexpected keyword argument 'profile_name'`.

- [ ] **Step 3: Thread the profile through**

In `agr_document_classifier/agr_document_classifier_classify.py`, change the import at line 24 to
`from utils.abc_embeddings import is_abc_embedding_model, get_profile, ABC_EMBEDDING_PROFILE, ABC_EMBEDDING_VERSION, ABC_EMBEDDING_DIM`,
then change the signature at lines 107-108 to:

```python
def classify_documents_from_abc_embeddings(reference_curies, mod_abbr, classifier_model, use_bow=False,
                                           embedding_cache=None,
                                           profile_name=ABC_EMBEDDING_PROFILE,
                                           version=ABC_EMBEDDING_VERSION):
```

Immediately after `bow_vectorizer = get_bow_vectorizer() if use_bow else None` (line 130), add:

```python
    # The zero row for a missing embedding must be as wide as this profile's
    # vectors. Unknown profiles fall back to the fulltext dim.
    profile = get_profile(profile_name, version)
    embedding_dim = profile.dim if profile else ABC_EMBEDDING_DIM
```

Replace the fetch-and-cache block (lines 133-138) with:

```python
        # Keyed on the profile too: the reclassify pipeline shares one cache
        # across models, and two profiles can share a dimension, so a
        # curie-only key would silently serve the wrong profile's vectors.
        cache_key = (reference_curie, profile_name, version)
        if embedding_cache is not None and cache_key in embedding_cache:
            result = embedding_cache[cache_key]
        else:
            result = get_reference_embedding(reference_curie, mod_abbr,
                                             profile_name=profile_name, version=version)
            if embedding_cache is not None:
                embedding_cache[cache_key] = result
```

Replace the zero-vector line (currently line 141) with:

```python
            pooled, text = np.zeros(embedding_dim, dtype=np.float32), ""
```

Update the docstring's `embedding_cache` paragraph (lines 123-127) to say the key is
`{(curie, profile, version): (pooled, text) | None}`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/agr_document_classifier/test_abstract_profile_plumbing.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `pytest tests/ -q && python3 -m flake8 .`
Expected: full suite passes — pay attention to `tests/agr_document_classifier/test_reclassify.py`, which exercises the shared cache.

```bash
git add agr_document_classifier/agr_document_classifier_classify.py \
        tests/agr_document_classifier/test_abstract_profile_plumbing.py
git commit -m "fix(SCRUM-5764): key embedding cache on profile and pass profile to fetch"
```

---

### Task 4: Read the profile from model metadata at the call sites

**Files:**
- Modify: `agr_document_classifier/agr_document_classifier_classify.py:287-300`
- Modify: `agr_document_classifier/agr_document_classifier_reclassify.py:159-160, 197-198`
- Test: `tests/agr_document_classifier/test_abstract_profile_plumbing.py`

**Interfaces:**
- Consumes: `classify_documents_from_abc_embeddings(..., profile_name=, version=)` (Task 3).
- Produces: `profile_pair_from_model(model_meta_data: Optional[dict]) -> Tuple[str, int]` in `utils/abc_embeddings.py` — returns the model's `(embedding_profile, embedding_version)`, defaulting to the fulltext pair when either is absent.

- [ ] **Step 1: Write the failing tests**

Append to `tests/agr_document_classifier/test_abstract_profile_plumbing.py`:

```python
from utils.abc_embeddings import profile_pair_from_model


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/agr_document_classifier/test_abstract_profile_plumbing.py -v`
Expected: FAIL — `ImportError: cannot import name 'profile_pair_from_model'`.

- [ ] **Step 3: Add the helper and use it at both call sites**

In `utils/abc_embeddings.py`, after `is_abc_embedding_model`, add:

```python
def profile_pair_from_model(model_meta_data: Optional[dict]) -> Tuple[str, int]:
    """Return the ``(embedding_profile, embedding_version)`` an ABC-embedding model
    was trained against, defaulting to the fulltext pair when the model does not
    carry one — which is every model uploaded before the abstract profile existed.
    """
    metadata = model_meta_data or {}
    profile_name = metadata.get("embedding_profile") or ABC_EMBEDDING_PROFILE
    version = metadata.get("embedding_version")
    return profile_name, ABC_EMBEDDING_VERSION if version is None else int(version)
```

In `agr_document_classifier_classify.py`, import `profile_pair_from_model` alongside the other
`utils.abc_embeddings` names, and in `process_job_batch` (line 287) derive the pair before the
call at line 298:

```python
        embedding_profile_name, embedding_version = profile_pair_from_model(model_meta_data)
        files_loaded, classifications, conf_scores, valid_embeddings = classify_documents_from_abc_embeddings(
            job_batch_curies, mod_abbr, classifier_model, use_bow=True,
            profile_name=embedding_profile_name, version=embedding_version)
```

Keep the existing positional arguments exactly as they are today; only add the two keyword
arguments. In `agr_document_classifier_reclassify.py`, do the same at both call sites (lines
159-160 and 197-198), taking the pair from the model metadata already loaded in that module and
passing `profile_name=` / `version=` alongside the existing `use_bow=True, embedding_cache=cache`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/agr_document_classifier/ -v`
Expected: PASS, including `test_reclassify.py`.

- [ ] **Step 5: Commit**

Run: `pytest tests/ -q && python3 -m flake8 .`

```bash
git add utils/abc_embeddings.py agr_document_classifier/agr_document_classifier_classify.py \
        agr_document_classifier/agr_document_classifier_reclassify.py \
        tests/agr_document_classifier/test_abstract_profile_plumbing.py
git commit -m "feat(SCRUM-5764): resolve embedding profile from model metadata"
```

---

### Task 5: Trainer `--embedding_profile`

**Files:**
- Modify: `agr_document_classifier/agr_document_classifier_trainer.py:93-128` (`_build_abc_embedding_features`), `:129-137` (`train_classifier` signature), `:468` area (CLI), `:635` area (wiring)
- Test: `tests/agr_document_classifier/test_abstract_profile_plumbing.py`

**Interfaces:**
- Consumes: `abc_embedding_recipe(profile_name, version)` (Task 1); profile-aware `get_reference_embedding` (Task 2).
- Produces: `_build_abc_embedding_features(abc_curies, mod_abbreviation, use_bow=False, profile_name=ABC_EMBEDDING_PROFILE, version=ABC_EMBEDDING_VERSION)`; CLI flag `--embedding_profile` accepting the profile *name* (default the fulltext name).

- [ ] **Step 1: Write the failing test**

Append to `tests/agr_document_classifier/test_abstract_profile_plumbing.py`:

```python
from agr_document_classifier import agr_document_classifier_trainer as trainer


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/agr_document_classifier/test_abstract_profile_plumbing.py::test_trainer_feature_build_uses_the_requested_profile -v`
Expected: FAIL — `TypeError: _build_abc_embedding_features() got an unexpected keyword argument 'profile_name'`.

- [ ] **Step 3: Thread the profile through the trainer**

Change the `_build_abc_embedding_features` signature (line 93) to:

```python
def _build_abc_embedding_features(abc_curies: dict, mod_abbreviation: str, use_bow: bool = False,
                                  profile_name: str = ABC_EMBEDDING_PROFILE,
                                  version: int = ABC_EMBEDDING_VERSION):
```

and its fetch (line 111) to:

```python
            result = get_reference_embedding(reference_curie, mod_abbreviation,
                                             profile_name=profile_name, version=version)
```

The trainer currently imports only `abc_embedding_recipe` from `utils.abc_embeddings`
(`agr_document_classifier_trainer.py:27`). Replace that line with:

```python
from utils.abc_embeddings import (abc_embedding_recipe, ABC_EMBEDDING_PROFILE,
                                  ABC_EMBEDDING_VERSION, ABSTRACT_PROFILE, FULLTEXT_PROFILE)
```

Add `embedding_profile: str = ABC_EMBEDDING_PROFILE` to `train_classifier`'s signature and pass
it into the `_build_abc_embedding_features` call at line 148; because a profile name identifies
exactly one registered version, look the version up with:

```python
        # A profile name identifies exactly one registered version today; resolve
        # it rather than making the operator pass a matching pair on the CLI.
        profile = next((p for p in (FULLTEXT_PROFILE, ABSTRACT_PROFILE)
                        if p.name == embedding_profile), None)
        if profile is None:
            raise ValueError(f"Unknown embedding profile: {embedding_profile}")
        X, y = _build_abc_embedding_features(abc_curies or {}, mod_abbreviation, use_bow=True,
                                             profile_name=profile.name, version=profile.version)
```

Add the CLI flag next to `--sections_to_use` (line 468):

```python
    parser.add_argument("--embedding_profile", type=str, default=ABC_EMBEDDING_PROFILE,
                        choices=[FULLTEXT_PROFILE.name, ABSTRACT_PROFILE.name],
                        help="Which ABC embedding profile to train on. Default is the fulltext "
                             "profile; the abstract profile trains on title+abstract only, for "
                             "classification that must run before a PDF exists.")
```

Pass `embedding_profile=args.embedding_profile` where `train_classifier` is invoked (line 635
area), and change the upload stamp so the model records the profile it was actually trained on:
`abc_embedding_recipe(profile_name=args.embedding_profile, version=profile.version)`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/agr_document_classifier/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `pytest tests/ -q && python3 -m flake8 .`

```bash
git add agr_document_classifier/agr_document_classifier_trainer.py \
        tests/agr_document_classifier/test_abstract_profile_plumbing.py
git commit -m "feat(SCRUM-5764): add --embedding_profile to the classifier trainer"
```

---

## Phase 2 — Training-set preparation

*No external blockers. Produces the numbers that size everything downstream.*

### Task 6: Prepare and audit ZFIN's labelled set

**Files:**
- Create: `scripts/prepare_zfin_probe_training_set.py`
- Input: Ceri's sheet exported as CSV (columns `AGRKBID`, `XREF`, `Classification`) — Google Sheet `1owwbLTlr_0tJiFkvTlpbFVI1Yg8LCjb6lO2iifet-KQ`, ~950 data rows
- Output: `<outdir>/labelled_abstracts.json`, `<outdir>/coverage_report.txt`

**Interfaces:**
- Consumes: `get_curie_from_xref` (`utils/abc_utils.py:534`), `get_reference_title_and_abstract` (`utils/abc_utils.py:647`).
- Produces: `labelled_abstracts.json` — a list of `{"curie": str, "xref": str, "label": "positive"|"negative", "title": str, "abstract": str}`, only for references that resolved **and** have a non-empty abstract. Consumed by Tasks 7 and 9.

**Why this task exists (do not skip the normalization):** the sheet's positive rows read `"Positive "` — capital P, **trailing space** — while the trainer compares `classification_value == "positive"` exactly (`agr_document_classifier_trainer.py:543-546`) and `add_entry_to_dataset` passes the value through unnormalized (`utils/abc_utils.py:1199`). Uploaded as-is, every row lands in neither the positive nor the negative list and training silently sees an empty set. There is also at least one duplicate row (`ZDB-PUB-260210-3`, Positive twice).

- [ ] **Step 1: Write the failing test**

Create `tests/agr_dataset_manager/test_prepare_zfin_probe_training_set.py`:

```python
import json
from unittest.mock import patch

from scripts import prepare_zfin_probe_training_set as prep


CSV = """AGRKBID,XREF,Classification
,ZDB-PUB-1,Positive 
,ZDB-PUB-1,Positive 
,ZDB-PUB-2,Negative
,ZDB-PUB-3,Negative
,ZDB-PUB-4,Positive 
"""


def test_normalizes_labels_dedups_and_drops_unusable_rows(tmp_path):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(CSV)

    # ZDB-PUB-3 does not resolve; ZDB-PUB-4 resolves but has no abstract.
    curies = {"ZDB-PUB-1": "AGRKB:1", "ZDB-PUB-2": "AGRKB:2",
              "ZDB-PUB-3": None, "ZDB-PUB-4": "AGRKB:4"}
    abstracts = {"AGRKB:1": ("Title one", "Abstract one."),
                 "AGRKB:2": ("Title two", "Abstract two."),
                 "AGRKB:4": ("Title four", "")}

    with patch.object(prep, "get_curie_from_xref", side_effect=lambda x: curies[x]), \
         patch.object(prep, "get_reference_title_and_abstract",
                      side_effect=lambda c: abstracts[c]):
        report = prep.prepare(str(csv_path), str(tmp_path))

    records = json.loads((tmp_path / "labelled_abstracts.json").read_text())
    # One positive (deduped) + one negative survive.
    assert [(r["curie"], r["label"]) for r in records] == [
        ("AGRKB:1", "positive"), ("AGRKB:2", "negative")]
    # Labels are lowercased and stripped — the trainer compares them exactly.
    assert all(r["label"] in ("positive", "negative") for r in records)
    assert report["rows_read"] == 5
    assert report["duplicates_dropped"] == 1
    assert report["unresolved_xrefs"] == 1
    assert report["missing_abstracts"] == 1
    assert report["usable"] == 2
    assert (tmp_path / "coverage_report.txt").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/agr_dataset_manager/test_prepare_zfin_probe_training_set.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.prepare_zfin_probe_training_set'`.

- [ ] **Step 3: Implement the script**

Create `scripts/__init__.py` if absent (empty file), then `scripts/prepare_zfin_probe_training_set.py`:

```python
"""Turn ZFIN's molecular-probe label sheet into an abstract-only training set
(SCRUM-5764).

The sheet (``AGRKBID``, ``XREF``, ``Classification``) is the format
``agr_dataset_manager/dataset_upload_from_csv.py`` already expects, but its label
values are ``"Positive "`` / ``"Negative"`` while the trainer compares
``classification_value == "positive"`` exactly. This script normalizes the labels,
drops duplicate xrefs, resolves each xref to an AGRKB curie, fetches title and
abstract, and reports how many references are actually usable — references with no
abstract in the ABC cannot be classified from an abstract at all, which caps
achievable recall and is the number ZFIN needs to size manual triage.
"""
import argparse
import csv
import json
import logging
import os
import sys

from utils.abc_utils import get_curie_from_xref, get_reference_title_and_abstract

logger = logging.getLogger(__name__)


def prepare(csv_file: str, output_dir: str) -> dict:
    """Write ``labelled_abstracts.json`` + ``coverage_report.txt`` into
    ``output_dir`` and return the coverage counters."""
    os.makedirs(output_dir, exist_ok=True)
    report = {"rows_read": 0, "unparseable_labels": 0, "duplicates_dropped": 0,
              "unresolved_xrefs": 0, "missing_abstracts": 0, "usable": 0}
    records = []
    seen_xrefs = set()

    with open(csv_file, newline="") as handle:
        for row in csv.DictReader(handle):
            report["rows_read"] += 1
            xref = (row.get("XREF") or "").strip()
            label = (row.get("Classification") or "").strip().lower()
            if not xref or label not in ("positive", "negative"):
                report["unparseable_labels"] += 1
                continue
            if xref in seen_xrefs:
                report["duplicates_dropped"] += 1
                continue
            seen_xrefs.add(xref)

            curie = (row.get("AGRKBID") or "").strip() or get_curie_from_xref(xref)
            if not curie:
                report["unresolved_xrefs"] += 1
                logger.warning("No AGRKB curie for xref %s", xref)
                continue

            title, abstract = get_reference_title_and_abstract(curie)
            if not (abstract or "").strip():
                report["missing_abstracts"] += 1
                logger.warning("No abstract for %s (%s)", curie, xref)
                continue

            records.append({"curie": curie, "xref": xref, "label": label,
                            "title": title or "", "abstract": abstract})
            report["usable"] += 1

    report["positives"] = sum(1 for r in records if r["label"] == "positive")
    report["negatives"] = sum(1 for r in records if r["label"] == "negative")

    with open(os.path.join(output_dir, "labelled_abstracts.json"), "w") as handle:
        json.dump(records, handle, indent=2)
    lines = [f"{key}: {value}" for key, value in report.items()]
    with open(os.path.join(output_dir, "coverage_report.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    logger.info("Coverage: %s", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--csv-file", required=True, help="Path to the exported label sheet")
    parser.add_argument("-o", "--output-dir", required=True, help="Where to write the outputs")
    parser.add_argument("-l", "--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, stream=sys.stdout,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    prepare(args.csv_file, args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/agr_dataset_manager/test_prepare_zfin_probe_training_set.py -v`
Expected: PASS.

- [ ] **Step 5: Run it for real against prod and read the coverage report**

```bash
set -a && . ./.env.rdsprod && set +a
python3 scripts/prepare_zfin_probe_training_set.py \
  -f /tmp/probe_training_set.csv -o /tmp/scrum5764
cat /tmp/scrum5764/coverage_report.txt
```

Expected: `rows_read: 950`, `duplicates_dropped >= 1`, and a `usable` count with its
positive/negative split. **Report these numbers before proceeding** — `missing_abstracts` is the
hard ceiling on recall and belongs in the ZFIN write-up.

- [ ] **Step 6: Commit**

Run: `pytest tests/ -q && python3 -m flake8 .`

```bash
git add scripts/prepare_zfin_probe_training_set.py scripts/__init__.py \
        tests/agr_dataset_manager/test_prepare_zfin_probe_training_set.py
git commit -m "feat(SCRUM-5764): add ZFIN probe training-set preparation script"
```

---

## Phase 3 — Abstract embedding generation

### Task 7: Generate ABC-format abstract embedding parquets

**Files:**
- Create: `scripts/generate_abstract_embeddings.py`
- Modify: `requirements.txt` (add `openai`)
- Test: `tests/agr_dataset_manager/test_generate_abstract_embeddings.py`

**Interfaces:**
- Consumes: `labelled_abstracts.json` (Task 6); `abstract_chunk_text`, `ABSTRACT_PROFILE` (Task 1).
- Produces: one `{curie}.parquet` per reference in the output directory, with columns `embedding` (`list<float32>`), `is_document_level` (`bool`), `content` (`string`) — the three columns `paragraph_pool_and_text` reads (`utils/abc_embeddings.py:100-105`) — plus `reference_curie`, `chunk_index`, `profile_name`, `chunking_strategy`, `section_title` to mirror the producer's schema (`embedding_generation.py:197-206`). Exactly one row, `is_document_level=False`, `chunk_index=0`.

**Note on `agr_abc_document_parsers`:** the producer uses `write_chunks_parquet` from
`agr_abc_document_parsers.embeddings`, which is **not present in the installed v1.5.1** (submodules
are `converter, jats_parser, md_emitter, md_reader, md_validator, models, plain_text, tei_parser,
xml_utils`). Write the three required columns with `pyarrow` directly for now, and switch to
`write_chunks_parquet` if the library is bumped to a version that ships it.

- [ ] **Step 1: Write the failing test**

Create `tests/agr_dataset_manager/test_generate_abstract_embeddings.py`:

```python
from unittest.mock import patch

import numpy as np

from scripts import generate_abstract_embeddings as gen
from utils.abc_embeddings import ABSTRACT_PROFILE, paragraph_pool_and_text


def test_parquet_round_trips_through_the_production_reader(tmp_path):
    # The whole point of writing ABC-format parquets is that the real consumer
    # reads them unchanged, so assert against paragraph_pool_and_text itself.
    vector = [0.0, 3.0, 4.0]
    with patch.object(gen, "embed_texts", return_value=[vector]):
        written = gen.generate(
            [{"curie": "AGRKB:1", "label": "positive",
              "title": "Probe synthesis", "abstract": "We made a probe."}],
            str(tmp_path))

    assert written == 1
    parquet_bytes = (tmp_path / "AGRKB:1.parquet").read_bytes()
    pooled, text = paragraph_pool_and_text(parquet_bytes)
    # A single chunk pools to its own L2-normalized vector: [0,3,4]/5.
    np.testing.assert_allclose(pooled, np.array([0.0, 0.6, 0.8], dtype=np.float32), rtol=1e-5)
    # The BoW block hashes exactly this text.
    assert text == "Probe synthesis\n\nWe made a probe."


def test_row_is_a_chunk_not_document_level(tmp_path):
    # A document-level row is skipped by the reader, which would make the
    # parquet unusable (paragraph_pool_and_text returns None).
    import pyarrow.parquet as pq

    with patch.object(gen, "embed_texts", return_value=[[1.0, 0.0]]):
        gen.generate([{"curie": "AGRKB:2", "label": "negative",
                       "title": "T", "abstract": "A"}], str(tmp_path))

    table = pq.read_table(tmp_path / "AGRKB:2.parquet")
    assert table.column("is_document_level").to_pylist() == [False]
    assert table.column("chunk_index").to_pylist() == [0]
    assert table.column("profile_name").to_pylist() == [ABSTRACT_PROFILE.name]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/agr_dataset_manager/test_generate_abstract_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.generate_abstract_embeddings'`.

- [ ] **Step 3: Add the dependency and implement the generator**

Add `openai` to `requirements.txt` (alphabetical position, unpinned minor: `openai==1.109.1`), then
create `scripts/generate_abstract_embeddings.py`:

```python
"""Embed title+abstract for a labelled reference set and write ABC-format
embedding parquets (SCRUM-5764).

One parquet per reference, one chunk row per parquet: an abstract is a single
~350-token chunk, so the L2-normalized chunk-mean pool degenerates to the
L2-normalized abstract vector. The ``content`` column carries the exact string
from ``utils.abc_embeddings.abstract_chunk_text`` — that is what the hashed BoW
block is built from, so it must match the ABC producer byte-for-byte.

Registration is deliberately not done here: ``embedding_file`` rows cannot be
created over HTTP (``embedding_file_router.py`` is read-only by design), so the
catalog row must come from the ABC-internal producer. See the plan's external
dependencies.
"""
import argparse
import json
import logging
import os
import sys
from typing import Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq

from utils.abc_embeddings import ABSTRACT_PROFILE, abstract_chunk_text

logger = logging.getLogger(__name__)

# Batch size for the embeddings endpoint. Abstracts are ~350 tokens, so this is
# well inside the request limit and keeps the run to a handful of calls.
_EMBED_BATCH = 100


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return one embedding per input text, in order, via the OpenAI embeddings
    API. Split out so tests can patch it without touching the network."""
    from openai import OpenAI

    client = OpenAI()
    vectors: List[List[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start:start + _EMBED_BATCH]
        response = client.embeddings.create(model=ABSTRACT_PROFILE.model_name, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        logger.info("Embedded %d/%d", min(start + _EMBED_BATCH, len(texts)), len(texts))
    return vectors


def _parquet_bytes(reference_curie: str, vector: List[float], content: str) -> bytes:
    table = pa.table({
        "embedding": pa.array([vector], type=pa.list_(pa.float32())),
        "is_document_level": pa.array([False], type=pa.bool_()),
        "content": pa.array([content], type=pa.string()),
        "reference_curie": pa.array([reference_curie], type=pa.string()),
        "chunk_index": pa.array([0], type=pa.int32()),
        "profile_name": pa.array([ABSTRACT_PROFILE.name], type=pa.string()),
        "chunking_strategy": pa.array(["abstract"], type=pa.string()),
        "section_title": pa.array(["__abstract__"], type=pa.string()),
    })
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer)
    return buffer.getvalue().to_pybytes()


def generate(records: Iterable[dict], output_dir: str) -> int:
    """Embed each record's title+abstract and write ``{curie}.parquet`` into
    ``output_dir``. Returns the number of parquets written."""
    os.makedirs(output_dir, exist_ok=True)
    records = list(records)
    texts = [abstract_chunk_text(r.get("title", ""), r.get("abstract", "")) for r in records]
    vectors = embed_texts(texts)
    for record, content, vector in zip(records, texts, vectors):
        path = os.path.join(output_dir, f"{record['curie']}.parquet")
        with open(path, "wb") as handle:
            handle.write(_parquet_bytes(record["curie"], vector, content))
    logger.info("Wrote %d parquet(s) to %s", len(records), output_dir)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-json", required=True,
                        help="labelled_abstracts.json from prepare_zfin_probe_training_set.py")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("-l", "--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, stream=sys.stdout,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with open(args.input_json) as handle:
        generate(json.load(handle), args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/agr_dataset_manager/test_generate_abstract_embeddings.py -v`
Expected: PASS.

- [ ] **Step 5: Generate for real**

```bash
export OPENAI_API_KEY=...   # ask the user; do not hardcode
python3 scripts/generate_abstract_embeddings.py \
  -i /tmp/scrum5764/labelled_abstracts.json -o /tmp/scrum5764/parquets
ls /tmp/scrum5764/parquets | wc -l
```

Expected: one parquet per usable reference. Cost at ~350 tokens each and
`text-embedding-3-small` ($0.02/1M) is under one cent for the whole set.

- [ ] **Step 6: Commit**

Run: `pytest tests/ -q && python3 -m flake8 .`

```bash
git add scripts/generate_abstract_embeddings.py requirements.txt \
        tests/agr_dataset_manager/test_generate_abstract_embeddings.py
git commit -m "feat(SCRUM-5764): generate ABC-format abstract embedding parquets"
```

---

### Task 8 (EXTERNALLY BLOCKED): Upload and register the parquets in ABC

**Do not start this task until external dependencies 2 and 3 are resolved.** Uploading the
parquet is possible today via `POST /reference/referencefile/file_upload/`
(`referencefile_router.py:42-56`, params `reference_curie`, `display_name`, `file_class`,
`file_publication_status`, `file_extension` + `file`), using `file_class="embedding"` and
`file_extension="parquet"`. But **without an `embedding_file` catalog row the upload is inert**:
`show_all` will report no `profile_name`, so `get_reference_embedding` skips it (Task 2's filter).

Resolve first, with Shuai Weng:
- Who creates the catalog row for a **source-less** embedding? The stated reason creation is not
  HTTP-exposed is that internal creation "enforces that the parquet inherits the source
  referencefile's access" — which has no meaning when `source_referencefile_id` is NULL. So either
  the producer job emits these, or a creation route is added for source-less profiles.
- What governs MOD download access on a source-less parquet?
- Confirm the profile name `classifier_abstract_title_abstract_single_chunk` and the exact
  `abstract_chunk_text` format, so a locally trained model finds matching features in production.

**Target environment: dev, not prod.** Writing eval artifacts to prod creates permanent
referencefiles under a profile name that is not yet settled, and an embedding is only removable by
deleting its parquet referencefile. Precedent for the split already exists: `ABC_UPLOAD_API_SERVER`
(`utils/abc_utils.py:1093-1094`) lets a run read data from one environment and write to another.
Before committing to dev, spot-check ~20 of the ZFIN curies from Task 6 actually exist there —
`embedding_file.reference_id` is `NOT NULL`, and most of these references are from 2025-2026.

**Until this is unblocked, Phase 4 reads the Task 7 parquets from disk** via
`paragraph_pool_and_text`, which is the same function the production path calls on downloaded
bytes — so the features are identical either way.

---

## Phase 4 — Three-arm evaluation

### Task 9: Evaluation harness

**Files:**
- Create: `scripts/evaluate_probe_classifiers.py`
- Test: `tests/agr_document_classifier/test_evaluate_probe_classifiers.py`

**Interfaces:**
- Consumes: `labelled_abstracts.json` (Task 6); `{curie}.parquet` files (Task 7); `paragraph_pool_and_text` (`utils/abc_embeddings.py`); `get_bow_vectorizer` (`utils/embedding.py:60`); `remove_stopwords` (`utils/get_documents.py:58`); `_select_and_fit_model` (`agr_document_classifier_trainer.py`).
- Produces: `build_matrix(records, parquet_dir, use_embedding: bool, use_bow: bool) -> Tuple[sp.csr_matrix, List[int], List[dict]]` — the third element is the records actually kept (those with a readable parquet), so row order and record order stay aligned; `holdout_indices(y) -> Tuple[np.ndarray, np.ndarray]`; `run_supervised(records, parquet_dir, use_embedding, use_bow) -> dict`; `evaluate(records, parquet_dir, outdir) -> dict` writing `results.json` and `results.md`.

**Split discipline — the arms must share one test set.** `_select_and_fit_model` performs its *own*
80/20 stratified holdout internally (`agr_document_classifier_trainer.py:215-217`, `test_size=0.20,
stratify=y, random_state=42`) and reports `stats["test_precision"/"test_recall"/"test_f1"]` from it.
So do **not** split before calling it — that would nest a second split inside the first, fitting on
60% of the data and producing two conflicting sets of "test" numbers. Instead reproduce that exact
deterministic split externally via `holdout_indices` to obtain probabilities and to select the LLM
arm's test records, so all three arms are scored on the identical references.

**Arms (fixed by the design discussion — do not add an embeddings-only arm):**
1. `embedding+bow` — the only supervised ship candidate.
2. `bow_only` — **diagnostic only.** If it matches arm 1, a shippable model needs no OpenAI key, no new profile, and no SCRUM-6140 dependency. That is a materially different operational story, which is why it is measured rather than assumed. (SCRUM-6052 found embeddings alone *underperformed* BoW on fulltext, and embedding+BoW merely matched it.)
3. `llm` — `gpt-5.4-nano`, Ceri's definition + few-shot drawn from the train split only.

- [ ] **Step 1: Write the failing test**

Create `tests/agr_document_classifier/test_evaluate_probe_classifiers.py`:

```python
import numpy as np
import scipy.sparse as sp

from scripts import evaluate_probe_classifiers as ev
from scripts import generate_abstract_embeddings as gen
from unittest.mock import patch


def _records():
    return [{"curie": f"AGRKB:{i}", "label": "positive" if i % 2 else "negative",
             "title": f"Title {i}", "abstract": f"Abstract number {i} about probes."}
            for i in range(1, 7)]


def _parquets(tmp_path, records):
    for i, record in enumerate(records):
        with patch.object(gen, "embed_texts", return_value=[[float(i), 1.0, 0.0]]):
            gen.generate([record], str(tmp_path))


def test_embedding_bow_matrix_is_sparse_and_wider_than_bow_alone(tmp_path):
    records = _records()
    _parquets(tmp_path, records)

    X_both, y_both, kept_both = ev.build_matrix(
        records, str(tmp_path), use_embedding=True, use_bow=True)
    X_bow, y_bow, kept_bow = ev.build_matrix(
        records, str(tmp_path), use_embedding=False, use_bow=True)

    assert sp.issparse(X_both) and sp.issparse(X_bow)
    assert X_both.shape[0] == X_bow.shape[0] == len(records)
    # The dense embedding block adds exactly its own width (3 here) on top of BoW.
    assert X_both.shape[1] == X_bow.shape[1] + 3
    assert y_both == y_bow == [0, 1, 0, 1, 0, 1]
    assert kept_both == kept_bow == records


def test_build_matrix_skips_references_with_no_parquet_and_reports_kept(tmp_path):
    records = _records()
    _parquets(tmp_path, records[:4])   # last two have no parquet
    X, y, kept = ev.build_matrix(records, str(tmp_path), use_embedding=True, use_bow=True)
    assert X.shape[0] == 4
    assert len(y) == 4
    # Row i must correspond to kept[i], or the LLM arm would score different
    # references than the supervised arms.
    assert [r["curie"] for r in kept] == ["AGRKB:1", "AGRKB:2", "AGRKB:3", "AGRKB:4"]


def test_holdout_indices_reproduce_the_trainer_internal_split():
    # _select_and_fit_model splits 80/20 stratified with random_state=42. The
    # arms are only comparable if we can reproduce exactly that test set.
    from sklearn.model_selection import train_test_split as sk_split

    y = np.array([0, 1] * 25)
    train_idx, test_idx = ev.holdout_indices(y)
    expected_train, expected_test = sk_split(
        np.arange(len(y)), test_size=0.20, stratify=y, random_state=42)
    np.testing.assert_array_equal(train_idx, expected_train)
    np.testing.assert_array_equal(test_idx, expected_test)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/agr_document_classifier/test_evaluate_probe_classifiers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.evaluate_probe_classifiers'`.

- [ ] **Step 3: Implement `build_matrix` and the supervised arms**

Create `scripts/evaluate_probe_classifiers.py`:

```python
"""Three-arm evaluation for ZFIN molecular-probe abstract classification
(SCRUM-5764).

Arms, on one fixed stratified split:

* ``embedding+bow`` — the supervised ship candidate. There is deliberately no
  embeddings-only arm: BoW is always concatenated (SCRUM-5781 decision 2, and
  ``_build_abc_embedding_features`` forces it).
* ``bow_only`` — diagnostic. If it matches ``embedding+bow``, a shippable model
  needs no OpenAI key and no abstract embedding profile at all.
* ``llm`` — a small chat model reading the abstract with the curator's
  definition, no training data.

Features are built with the production reader (``paragraph_pool_and_text``) over
the parquets, so what is measured here is what would ship.
"""
import argparse
import json
import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from utils.abc_embeddings import paragraph_pool_and_text
from utils.embedding import get_bow_vectorizer
from utils.get_documents import remove_stopwords

logger = logging.getLogger(__name__)


def holdout_indices(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the 80/20 stratified holdout ``_select_and_fit_model`` performs
    internally (``test_size=0.20, stratify=y, random_state=42``) so every arm is
    scored on exactly the same references."""
    return train_test_split(np.arange(len(y)), test_size=0.20, stratify=y, random_state=42)


def build_matrix(records: List[dict], parquet_dir: str, use_embedding: bool,
                 use_bow: bool) -> Tuple[sp.csr_matrix, List[int], List[dict]]:
    """Build ``(X, y, kept_records)`` from the abstract embedding parquets, mirroring
    ``_build_abc_embedding_features``: L2-normalized chunk-mean pool for the dense
    block, hashed BoW over the parquet's own ``content`` for the sparse block.

    References with no readable parquet are skipped, so ``kept_records`` is returned
    to keep row order aligned with record order — the LLM arm scores the same rows.
    """
    if not (use_embedding or use_bow):
        raise ValueError("At least one of use_embedding / use_bow must be set")
    bow_vectorizer = get_bow_vectorizer() if use_bow else None
    rows, y, kept = [], [], []
    for record in records:
        path = os.path.join(parquet_dir, f"{record['curie']}.parquet")
        if not os.path.exists(path):
            logger.warning("No parquet for %s; skipping", record["curie"])
            continue
        with open(path, "rb") as handle:
            result = paragraph_pool_and_text(handle.read())
        if result is None:
            logger.warning("Unreadable parquet for %s; skipping", record["curie"])
            continue
        pooled, text = result
        blocks = []
        if use_embedding:
            blocks.append(sp.csr_matrix(pooled.reshape(1, -1)))
        if use_bow:
            blocks.append(bow_vectorizer.transform([remove_stopwords(text).lower() if text else ""]))
        rows.append(sp.hstack(blocks, format="csr"))
        y.append(int(record["label"] == "positive"))
        kept.append(record)
    return sp.vstack(rows, format="csr"), y, kept


def _scores(y_true, y_pred, y_prob) -> dict:
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_test": int(len(y_true)),
        "n_positive_test": int(sum(y_true)),
    }
    if y_prob is not None:
        out["average_precision"] = float(average_precision_score(y_true, y_prob))
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        # The operational question: how much recall survives at the precision
        # ZFIN needs before auto-setting "won't curate"?
        for target in (0.90, 0.95, 0.99):
            usable = [(p, r, t) for p, r, t in zip(precision, recall, thresholds) if p >= target]
            best = max(usable, key=lambda item: item[1]) if usable else None
            out[f"recall_at_precision_{int(target * 100)}"] = (
                {"recall": float(best[1]), "threshold": float(best[2])} if best else None)
    return out
```

Then add the supervised runner, reusing the trainer's model selection so the winning arm is
already production-shaped:

```python
def run_supervised(records, parquet_dir, use_embedding, use_bow) -> dict:
    """Fit via the trainer's own model selection and score on its internal holdout.

    ``_select_and_fit_model`` is handed the full matrix on purpose: it performs the
    80/20 stratified split itself. ``holdout_indices`` reproduces that split so we
    can pull probabilities for the precision/recall-threshold analysis and so the
    LLM arm can be scored on the same references.
    """
    from agr_document_classifier.agr_document_classifier_trainer import _select_and_fit_model

    X, y_list, kept = build_matrix(records, parquet_dir,
                                   use_embedding=use_embedding, use_bow=use_bow)
    y = np.array(y_list)
    model, stats = _select_and_fit_model(X, y, False, "isolation_forest", 0.1,
                                        use_bow_features=use_bow, use_lsh_features=False)

    _train_idx, test_idx = holdout_indices(y)
    X_test, y_test = X[test_idx], y[test_idx]
    y_pred = model.predict(X_test)
    # LinearSVC has no predict_proba; the production classify path falls back to
    # decision_function -> sigmoid, so mirror that here for the PR curve.
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = 1.0 / (1.0 + np.exp(-model.decision_function(X_test)))
    else:
        y_prob = None

    result = _scores(y_test, y_pred, y_prob)
    result["selected_model"] = stats.get("model_name")
    result["cv_f1"] = stats.get("average_f1")
    result["n_features"] = int(X.shape[1])
    result["n_records_used"] = len(kept)
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/agr_document_classifier/test_evaluate_probe_classifiers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `pytest tests/ -q && python3 -m flake8 .`

```bash
git add scripts/evaluate_probe_classifiers.py \
        tests/agr_document_classifier/test_evaluate_probe_classifiers.py
git commit -m "feat(SCRUM-5764): add supervised arms of the probe evaluation harness"
```

---

### Task 10: LLM arm

**Files:**
- Modify: `scripts/evaluate_probe_classifiers.py`
- Test: `tests/agr_document_classifier/test_evaluate_probe_classifiers.py`

**Interfaces:**
- Consumes: `_scores` and the train/test split from Task 9.
- Produces: `classify_with_llm(record: dict, few_shot: List[dict], model: str) -> Tuple[int, float]` returning `(label, confidence)`; `run_llm(test_records: List[dict], few_shot: List[dict], model: str) -> dict`; `_chat_json(prompt: str, model: str) -> str` (the network seam tests patch).

- [ ] **Step 1: Write the failing test**

Append to `tests/agr_document_classifier/test_evaluate_probe_classifiers.py`:

```python
def test_llm_arm_parses_structured_verdicts():
    responses = ['{"label": "positive", "confidence": 0.91}',
                 '{"label": "negative", "confidence": 0.12}']
    with patch.object(ev, "_chat_json", side_effect=responses):
        first = ev.classify_with_llm(
            {"title": "Synthesis of a probe", "abstract": "We characterize a new dye."},
            few_shot=[], model="gpt-5.4-nano")
        second = ev.classify_with_llm(
            {"title": "Gene X in heart development", "abstract": "We used a probe to stain."},
            few_shot=[], model="gpt-5.4-nano")
    assert first == (1, 0.91)
    assert second == (0, 0.12)


def test_llm_arm_treats_unparseable_output_as_negative_low_confidence():
    # A malformed verdict must not crash a 950-abstract run, and must not be
    # silently counted as a positive.
    with patch.object(ev, "_chat_json", return_value="not json at all"):
        assert ev.classify_with_llm({"title": "T", "abstract": "A"},
                                    few_shot=[], model="gpt-5.4-nano") == (0, 0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/agr_document_classifier/test_evaluate_probe_classifiers.py -v`
Expected: FAIL — `AttributeError: module 'scripts.evaluate_probe_classifiers' has no attribute 'classify_with_llm'`.

- [ ] **Step 3: Implement the LLM arm**

Append to `scripts/evaluate_probe_classifiers.py`:

```python
# Ceri Van Slyke's class definition, from SCRUM-5764. The "only" is the whole
# difficulty: a paper that develops a probe AND reports zebrafish biology is a
# negative, because it still has curatable data.
_SYSTEM_PROMPT = """You classify zebrafish literature for ZFIN curators, using only the title and abstract.

Label a reference POSITIVE only if the paper is solely about the synthesis, development, or
characterization of a molecular probe (for example a colorimetric, radiometric, or fluorescent
probe) and reports no other curatable zebrafish biology.

Label it NEGATIVE if it reports any other curatable finding — gene function, expression,
phenotype, disease modelling, development — even when a probe is used as a tool, and even when the
word "probe" appears in the abstract.

Reply with JSON only: {"label": "positive"|"negative", "confidence": <0.0-1.0>}"""


def _chat_json(prompt: str, model: str) -> str:
    """Send one classification request and return the raw response text. Split out
    so tests can patch it without touching the network."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _prompt_for(record: dict, few_shot: List[dict]) -> str:
    parts = []
    for example in few_shot:
        parts.append(f"Title: {example['title']}\nAbstract: {example['abstract']}\n"
                     f"Answer: {{\"label\": \"{example['label']}\", \"confidence\": 1.0}}")
    parts.append(f"Title: {record.get('title', '')}\nAbstract: {record.get('abstract', '')}\nAnswer:")
    return "\n\n".join(parts)


def classify_with_llm(record: dict, few_shot: List[dict], model: str) -> Tuple[int, float]:
    """Return ``(label, confidence)`` for one reference. Unparseable output is
    treated as a low-confidence negative so one bad verdict cannot abort a full
    run or inflate the positive count."""
    raw = _chat_json(_prompt_for(record, few_shot), model)
    try:
        verdict = json.loads(raw)
        label = 1 if str(verdict["label"]).strip().lower() == "positive" else 0
        return label, float(verdict.get("confidence", 0.0))
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Unparseable LLM verdict %r: %s", raw, exc)
        return 0, 0.0


def run_llm(test_records: List[dict], few_shot: List[dict], model: str) -> dict:
    y_true, y_pred, y_prob = [], [], []
    for record in test_records:
        label, confidence = classify_with_llm(record, few_shot, model)
        y_true.append(int(record["label"] == "positive"))
        y_pred.append(label)
        # Signed confidence so the PR curve is meaningful: a confident negative
        # must rank below an unsure positive.
        y_prob.append(confidence if label == 1 else 1.0 - confidence)
    result = _scores(np.array(y_true), np.array(y_pred), np.array(y_prob))
    result["model"] = model
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/agr_document_classifier/test_evaluate_probe_classifiers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `pytest tests/ -q && python3 -m flake8 .`

```bash
git add scripts/evaluate_probe_classifiers.py \
        tests/agr_document_classifier/test_evaluate_probe_classifiers.py
git commit -m "feat(SCRUM-5764): add LLM arm to the probe evaluation harness"
```

---

### Task 11: Run the evaluation and write the ZFIN report

**Files:**
- Modify: `scripts/evaluate_probe_classifiers.py` (add `evaluate` + `main`)
- Create: `docs/superpowers/reports/2026-08-06-scrum-5764-evaluation-results.md`

**Interfaces:**
- Consumes: `run_supervised` (Task 9), `run_llm` (Task 10).
- Produces: `results.json` + `results.md` in the output dir; a committed report.

- [ ] **Step 1: Add the driver**

Append to `scripts/evaluate_probe_classifiers.py`:

```python
def evaluate(records: List[dict], parquet_dir: str, output_dir: str,
             llm_model: str = "gpt-5.4-nano", n_few_shot: int = 8) -> dict:
    """Run all three arms on one shared holdout and write the results."""
    os.makedirs(output_dir, exist_ok=True)

    # Establish the shared split from the same kept-records list the supervised
    # arms use, so the LLM arm scores exactly the same references. Both supervised
    # arms keep the same records (each needs a readable parquet), so one call is
    # enough to derive the split.
    _X, y_list, kept = build_matrix(records, parquet_dir, use_embedding=True, use_bow=True)
    y = np.array(y_list)
    train_idx, test_idx = holdout_indices(y)
    train_records = [kept[i] for i in train_idx]
    test_records = [kept[i] for i in test_idx]

    # Few-shot examples come only from the train side, balanced, so the LLM arm
    # never sees a test abstract.
    positives = [r for r in train_records if r["label"] == "positive"][:n_few_shot // 2]
    negatives = [r for r in train_records if r["label"] == "negative"][:n_few_shot // 2]

    results = {
        "n_records_input": len(records),
        "n_records_used": len(kept),
        "n_train": len(train_records),
        "n_test": len(test_records),
        "arms": {
            "embedding+bow": run_supervised(records, parquet_dir, True, True),
            "bow_only": run_supervised(records, parquet_dir, False, True),
            "llm": run_llm(test_records, positives + negatives, llm_model),
        },
    }
    with open(os.path.join(output_dir, "results.json"), "w") as handle:
        json.dump(results, handle, indent=2)

    lines = ["| arm | precision | recall | F1 | AP |", "| --- | --: | --: | --: | --: |"]
    for name, scores in results["arms"].items():
        lines.append(f"| {name} | {scores['precision']:.3f} | {scores['recall']:.3f} | "
                     f"{scores['f1']:.3f} | {scores.get('average_precision', float('nan')):.3f} |")
    with open(os.path.join(output_dir, "results.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    logger.info("Results written to %s", output_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-json", required=True)
    parser.add_argument("-p", "--parquet-dir", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--llm-model", default="gpt-5.4-nano")
    parser.add_argument("-l", "--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, stream=sys.stdout,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with open(args.input_json) as handle:
        evaluate(json.load(handle), args.parquet_dir, args.output_dir, llm_model=args.llm_model)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full evaluation on cervino**

cervino has 62 GB RAM / 51 GB available / 16 cores. The matrix is ~950 × 263,680 in CSR
(~20 MB); peak memory is the `RandomizedSearchCV` workers (XGBoost histograms ~3 GB/fit, capped at
`SEARCH_MAX_JOBS=4`), so ~12 GB. No BioWordVec is loaded on this path. If memory is tight, set
`SEARCH_MAX_JOBS=2`.

```bash
ssh cervino.caltech.edu
cd ~/workspace/agr/agr_automated_information_extraction   # adjust to the checkout
conda activate agr_automated_information_extraction
export OPENAI_API_KEY=...
SEARCH_MAX_JOBS=4 python3 scripts/evaluate_probe_classifiers.py \
  -i /tmp/scrum5764/labelled_abstracts.json \
  -p /tmp/scrum5764/parquets \
  -o /tmp/scrum5764/results
cat /tmp/scrum5764/results/results.md
```

Expected: three rows of metrics. LLM arm cost for ~240 test abstracts is a few cents.

- [ ] **Step 3: Write the report**

Create `docs/superpowers/reports/2026-08-06-scrum-5764-evaluation-results.md` containing: the
coverage numbers from Task 6 (especially `missing_abstracts`, the recall ceiling), the three-arm
metrics table, the `recall_at_precision_95` figure and its threshold as the recommended
auto-"won't curate" gate, whether `bow_only` matched `embedding+bow` (and therefore whether the
abstract embedding profile is needed at all), and performance on the hard negatives Ceri
annotated `string "probe" in abstract`.

- [ ] **Step 4: Commit**

Run: `pytest tests/ -q && python3 -m flake8 .`

```bash
git add scripts/evaluate_probe_classifiers.py \
        docs/superpowers/reports/2026-08-06-scrum-5764-evaluation-results.md
git commit -m "feat(SCRUM-5764): run three-arm probe classification evaluation"
```

---

## Out of scope (follow-up plan once blockers clear)

- Creating the "molecular probe" ATP term (Ceri) and the dependent ATP tags for the ZFIN
  workflow action: "won't manually index" `ATP:0000343`, "low priority data" `ATP:0000322`,
  "won't curate" `ATP:0000299`.
- Uploading the labelled set to ABC as a dataset (`agr_dataset_manager/dataset_upload_from_csv.py`
  — feed it the *normalized* CSV from Task 6) and training/registering a production model.
- The ABC-side abstract embedding pipeline and its trigger point (SCRUM-6140). **Raise with Ceri:**
  the ticket says "embed abstract as soon as papers made inside corpus", but the ZFIN acquisition
  flow needs the tag set at the biblio-only stage, right after the PubMed acquisition script — so
  the trigger may need to be reference creation rather than `inside_corpus`.
- MeSH-derived features as a third feature block.
- SCRUM-5765 (ZFIN protocol papers) reusing this harness — same shape, and it would reuse Tasks
  1-5 unchanged plus a new profile-free label set.
- Adding a `write_chunks_parquet`-based writer once `agr_abc_document_parsers` ships
  `.embeddings`.
