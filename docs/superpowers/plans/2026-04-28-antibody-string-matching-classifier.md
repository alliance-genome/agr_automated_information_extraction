# WB Antibody String-Matching Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate WB's caltech antibody string-matching pipeline to ABC, emitting TopicEntityTags (positive/negative + matched-span notes) instead of WB DB rows, with workflow tags driving job orchestration.

**Architecture:** New string-matching classifier under `agr_document_classifier/` that pulls jobs from ABC via the workflow-tag mechanism, runs three regex rule sets (anti-GENE, combinations, additional keywords) over the TEI full text, and POSTs TET tags. Workflow trigger and backfill are handled by two SCRUM-named oneoff scripts in `agr_literature_service`.

**Tech Stack:** Python 3.12, pytest, regex (`re`), `agr_curation_api` (gene fetch), `AllianceTEI` (TEI parsing), `urllib`/`requests` (ABC API).

**Spec:** `docs/superpowers/specs/2026-04-27-antibody-string-matching-classifier-design.md`

---

## File structure

### `agr_automated_information_extraction` (branch `SCRUM-5601`)

| Path | Responsibility |
|---|---|
| `agr_document_classifier/antibody_rules.py` (new) | Pure regex rule engine: build regexes, match sentences. No I/O. |
| `agr_document_classifier/agr_antibody_string_matching_classifier.py` (new) | Pipeline entry point: load jobs, download TEIs, apply rules, POST TETs. |
| `utils/abc_utils.py` (modify) | Add `note` param to `send_classification_tag_to_abc`; add eco_map entry for new source. |
| `tests/agr_document_classifier/__init__.py` (new) | Empty — make the test dir a package. |
| `tests/agr_document_classifier/test_antibody_rules.py` (new) | Unit tests for rule engine. |
| `tests/agr_document_classifier/test_antibody_classifier_pipeline.py` (new) | Integration tests for pipeline with mocks. |
| `tests/utils/__init__.py` (new if missing) | Empty package marker. |
| `tests/utils/test_abc_utils_note.py` (new) | Test that `note` field passes through `send_classification_tag_to_abc` payload. |
| `Makefile` (modify) | Add `classify_antibody` target. |

### `agr_literature_service` (branch `SCRUM-5601`)

| Path | Responsibility |
|---|---|
| `agr_literature_service/lit_processing/oneoff_scripts/SCRUM-5601_add_antibody_str_match_transitions.py` (new) | Idempotent: INSERT four transitions, UPDATE two text-conversion `actions` arrays. |
| `agr_literature_service/lit_processing/oneoff_scripts/SCRUM-5601_backfill_antibody_str_match_needed_tag.py` (new) | Idempotent: INSERT ATP:0000366 for WB in-corpus refs not covered by legacy caltech. |

---

## Task 1: Add `note` parameter to `send_classification_tag_to_abc`

**Files:**
- Modify: `utils/abc_utils.py:212-253`
- Create: `tests/utils/__init__.py`
- Create: `tests/utils/test_abc_utils_note.py`

- [ ] **Step 1.1: Create empty `tests/utils/__init__.py`**

```bash
mkdir -p /home/valerio/workspace/agr/agr_automated_information_extraction/tests/utils
touch /home/valerio/workspace/agr/agr_automated_information_extraction/tests/utils/__init__.py
```

- [ ] **Step 1.2: Write failing test for `note` field passthrough**

Create `tests/utils/test_abc_utils_note.py` with:

```python
import json
from unittest.mock import patch, MagicMock
from utils import abc_utils


def _read_payload_from_request(mock_request_cls):
    """Pull the JSON body out of the urlopen Request object the SUT built."""
    args, kwargs = mock_request_cls.call_args
    data = kwargs.get("data") or args[1]
    return json.loads(data.decode("utf-8"))


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_note_is_included_when_provided(mock_request, mock_urlopen, _tok):
    mock_urlopen.return_value.__enter__.return_value.getcode.return_value = 201

    abc_utils.send_classification_tag_to_abc(
        reference_curie="AGRKB:101000000000001",
        species="NCBITaxon:6239",
        topic="ATP:0000096",
        negated=False,
        data_novelty="ATP:0000335",
        confidence_score=None,
        confidence_level=None,
        tet_source_id=42,
        note="anti-PDR-1, raised antibody",
    )

    payload = _read_payload_from_request(mock_request)
    assert payload["note"] == "anti-PDR-1, raised antibody"


@patch("utils.abc_utils.get_authentication_token", return_value="t")
@patch("utils.abc_utils.urllib.request.urlopen")
@patch("utils.abc_utils.urllib.request.Request")
def test_note_is_omitted_when_none(mock_request, mock_urlopen, _tok):
    mock_urlopen.return_value.__enter__.return_value.getcode.return_value = 201

    abc_utils.send_classification_tag_to_abc(
        reference_curie="AGRKB:101000000000001",
        species="NCBITaxon:6239",
        topic="ATP:0000096",
        negated=True,
        data_novelty="ATP:0000335",
        confidence_score=None,
        confidence_level=None,
        tet_source_id=42,
        note=None,
    )

    payload = _read_payload_from_request(mock_request)
    assert "note" not in payload
```

- [ ] **Step 1.3: Run test — expect FAIL (note kwarg not yet supported)**

Run: `cd /home/valerio/workspace/agr/agr_automated_information_extraction && python -m pytest tests/utils/test_abc_utils_note.py -v`
Expected: both tests FAIL with `TypeError: send_classification_tag_to_abc() got an unexpected keyword argument 'note'`.

- [ ] **Step 1.4: Add `note` parameter and conditional payload key**

Edit `utils/abc_utils.py`. Replace the function signature and `tet_data` block (currently lines 212-230):

```python
def send_classification_tag_to_abc(reference_curie: str, species: str, topic: str, negated: bool,
                                   data_novelty: str, confidence_score: float,
                                   confidence_level: str, tet_source_id, ml_model_id: Optional[int] = None,
                                   note: Optional[str] = None):
    url = f'{blue_api_base_url}/topic_entity_tag/'
    token = get_authentication_token()
    payload = {
        "created_by": "default_user",
        "updated_by": "default_user",
        "topic": topic,
        "species": species,
        "topic_entity_tag_source_id": tet_source_id,
        "negated": negated,
        "data_novelty": data_novelty,
        "confidence_score": float(confidence_score) if confidence_score is not None else None,
        "confidence_level": confidence_level,
        "reference_curie": reference_curie,
        "ml_model_id": ml_model_id,
        "force_insertion": True,
    }
    if note is not None:
        payload["note"] = note
    tet_data = json.dumps(payload).encode('utf-8')
```

Note: also replaced the unconditional `float(confidence_score)` with a None-safe variant since D1 says confidence_score may be `None`.

- [ ] **Step 1.5: Run tests — expect PASS**

Run: `python -m pytest tests/utils/test_abc_utils_note.py -v`
Expected: both tests PASS.

- [ ] **Step 1.6: Commit**

```bash
cd /home/valerio/workspace/agr/agr_automated_information_extraction
git add utils/abc_utils.py tests/utils/__init__.py tests/utils/test_abc_utils_note.py
git commit -m "feat(SCRUM-5601): pass note through send_classification_tag_to_abc

The TET API already supports a note field on TopicEntityTag; this exposes
it from the Python helper so the antibody string-matching classifier can
record matched spans alongside the positive/negative tag. Also makes
confidence_score None-safe for deterministic classifiers that don't
report a numeric score.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add eco_map entry for the new source method

**Files:**
- Modify: `utils/abc_utils.py:131-138` (the `eco_map` dict in `get_tet_source_id`)

This is an N=1 dict-key addition — TDD doesn't add value here. Direct edit + manual verification.

- [ ] **Step 2.1: Add the new key**

In `utils/abc_utils.py`, locate the `eco_map` dict inside `get_tet_source_id` (currently around line 132) and add:

```python
    eco_map = {
        "abc_entity_extractor": "ECO:0008021",  # string matching
        "abc_document_classifier": "ECO:0008004",  # ML
        "abc_literature_system": "ATP:0000036",
        "abc_string_matching_antibody": "ECO:0008021",  # WB antibody string matching topic classifier
    }
```

- [ ] **Step 2.2: Verify the import surface still parses**

Run: `python -c "from utils import abc_utils; print(abc_utils.send_classification_tag_to_abc.__doc__ or 'ok')"`
Expected: prints `ok` (no SyntaxError, no ImportError).

- [ ] **Step 2.3: Commit**

```bash
git add utils/abc_utils.py
git commit -m "feat(SCRUM-5601): register abc_string_matching_antibody source method

Maps the new TET source method to the string-matching ECO code so
get_tet_source_id can either fetch the existing source row or POST a new
one on first invocation per environment.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Create the antibody rule engine (`antibody_rules.py`)

**Files:**
- Create: `agr_document_classifier/antibody_rules.py`
- Create: `tests/agr_document_classifier/__init__.py`
- Create: `tests/agr_document_classifier/test_antibody_rules.py`

- [ ] **Step 3.1: Create empty `tests/agr_document_classifier/__init__.py`**

```bash
mkdir -p /home/valerio/workspace/agr/agr_automated_information_extraction/tests/agr_document_classifier
touch /home/valerio/workspace/agr/agr_automated_information_extraction/tests/agr_document_classifier/__init__.py
```

- [ ] **Step 3.2: Write failing tests covering all three rule sets and the case-sensitivity filter**

Create `tests/agr_document_classifier/test_antibody_rules.py`:

```python
from agr_document_classifier.antibody_rules import build_regexes, match_antibody_spans


def _matches(sentences, curated_genes):
    rules = build_regexes(curated_gene_names=curated_genes)
    return match_antibody_spans(sentences, rules)


def test_anti_gene_with_capitalized_suffix_matches():
    out = _matches(["This study used anti-PDR-1 antibody."], ["pdr-1"])
    assert "anti-PDR-1" in out


def test_anti_gene_lowercase_suffix_filtered_out():
    """Caltech filter keeps only matches where the gene-name suffix has at least
    one capital letter. anti-pdr-1 in lowercase is dropped."""
    out = _matches(["See anti-pdr-1 protein expression."], ["pdr-1"])
    assert not any(m.startswith("anti-") for m in out)


def test_excluded_gene_pdi_never_matches():
    """PDI is in EXCLUDE_GENES; even if passed in, it should not appear in
    the anti-GENE regex alternatives."""
    out = _matches(["See anti-PDI antibody and anti-Pdi protein."], ["pdi"])
    assert not any("pdi" in m.lower() for m in out)


def test_anti_c_elegans_matches():
    out = _matches(["The anti-C. elegans serum was used."], [])
    assert "anti-C. elegans" in out


def test_anti_msp_matches_via_additional_anti_keyword():
    """MSP is in ADDITIONAL_ANTI_KEYWORDS; antibody_rules adds it to the
    anti-GENE alternatives."""
    out = _matches(["The anti-MSP antibody recognized the protein."], [])
    assert "anti-MSP" in out


def test_combination_raised_antibody():
    out = _matches(["The antibody was raised against UNC-54."], ["unc-54"])
    assert "raised antibody" in out


def test_combination_preparation_antibodies_either_order():
    """Combinations match in either order: '<comb1> ... <comb2>' or
    '<comb2> ... <comb1>'."""
    out = _matches(["preparation of monoclonal antibodies for the study."], [])
    assert "preparation antibodies" in out


def test_additional_keyword_mh46_matches():
    out = _matches(["The MH46 antibody was used in this study."], [])
    assert "MH46" in out


def test_additional_keyword_sp56_matches():
    out = _matches(["Loaded the a-SP56 reagent into the gel."], [])
    assert "a-SP56" in out


def test_no_rule_fires_no_match():
    out = _matches(["We bought a commercial antibody from Sigma."], ["unc-54"])
    assert out == set()


def test_en_dash_normalized_to_hyphen():
    """'anti–PDR-1' (en-dash) should be treated as 'anti-PDR-1'."""
    out = _matches(["Used the anti–PDR-1 antibody."], ["pdr-1"])
    assert "anti-PDR-1" in out


def test_multiple_matches_in_one_sentence():
    out = _matches(
        ["The anti-PDR-1 antibody was raised in rabbits."],
        ["pdr-1"],
    )
    assert "anti-PDR-1" in out
    assert "raised antibody" in out
```

- [ ] **Step 3.3: Run tests — expect FAIL (module not yet created)**

Run: `python -m pytest tests/agr_document_classifier/test_antibody_rules.py -v`
Expected: collection error or all tests fail with `ModuleNotFoundError: No module named 'agr_document_classifier.antibody_rules'`.

- [ ] **Step 3.4: Implement `agr_document_classifier/antibody_rules.py`**

Create the file with the full rule engine — ported verbatim from `caltech/entity-extraction-antibody/main.py:50-82` but exposed as pure functions:

```python
"""Pure-function antibody string-matching rule engine.

Ported from the Caltech WB antibody pipeline (entity-extraction-antibody/main.py).
The rules are MOD-agnostic by data; the caller supplies the curated gene list.

No I/O, no ABC imports — trivially unit-testable.
"""

import itertools
import re
from typing import Iterable, NamedTuple


EXCLUDE_GENES = ['PDI']
ADDITIONAL_ANTI_KEYWORDS = ['MSP']
ADDITIONAL_KEYWORDS = ['MH46', 'SP56', 'a-SP56']

COMBINATION_1 = [
    "preparation", "prepared", "prepare", "production", "purification",
    "generation", "generate", "generated", "produce", "produced",
    "purify", "purified", "raised",
]
COMBINATION_2 = ["antiserum", "antibody", "antibodies", "antisera"]


class AntibodyRules(NamedTuple):
    anti_gene_regex: re.Pattern
    additional_keywords_regex: re.Pattern
    combinations_regex: list  # list[tuple[(str, str), re.Pattern, re.Pattern]]


def build_regexes(curated_gene_names: Iterable[str]) -> AntibodyRules:
    """Compile the three rule sets once. Reuse across many sentences/papers.

    `curated_gene_names` is the WB gene list (lowercase or any case — we lowercase
    here). EXCLUDE_GENES are removed; ADDITIONAL_ANTI_KEYWORDS are appended.
    """
    excluded_lower = {g.lower() for g in EXCLUDE_GENES}
    gene_names = {g.lower() for g in curated_gene_names if g.lower() not in excluded_lower}
    gene_names.update(g.lower() for g in ADDITIONAL_ANTI_KEYWORDS)
    escaped_genes = sorted(re.escape(g) for g in gene_names)

    if escaped_genes:
        anti_gene_pattern = (
            r"(?i)[\s\(\[\{\.,;:\'\"\<](anti\-(?:"
            + "|".join(escaped_genes)
            + r"|C\. elegans))[\s\.;:,'\"\)\]\}\>\?]"
        )
    else:
        # No curated genes — only match anti-C. elegans
        anti_gene_pattern = (
            r"(?i)[\s\(\[\{\.,;:\'\"\<](anti\-C\. elegans)[\s\.;:,'\"\)\]\}\>\?]"
        )
    anti_gene_regex = re.compile(anti_gene_pattern)

    additional_keywords_regex = re.compile(
        r"[\s\(\[\{\.,;:\'\"\<]("
        + "|".join(re.escape(k) for k in ADDITIONAL_KEYWORDS)
        + r")[\s\.;:,'\"\)\]\}\>\?]"
    )

    combinations = list(itertools.product(COMBINATION_1, COMBINATION_2))
    combinations_regex = [
        (
            comb,
            re.compile(".*" + comb[0] + r"\s.*" + comb[1] + r"[\s\.;:,'\"\)\]\}\>\?]"),
            re.compile(".*" + comb[1] + r"\s.*" + comb[0] + r"[\s\.;:,'\"\)\]\}\>\?]"),
        )
        for comb in combinations
    ]

    return AntibodyRules(anti_gene_regex, additional_keywords_regex, combinations_regex)


def _normalize(sentence: str) -> str:
    """Replace various Unicode dashes with a plain hyphen so regexes hit them."""
    return sentence.replace('–', '-').replace('‐', '-')


def match_antibody_spans(sentences: Iterable[str], rules: AntibodyRules) -> set:
    """Apply all three rule sets to a list of sentences. Return the union of
    matched span strings.

    The case-sensitivity filter on anti-GENE matches requires at least one
    uppercase character in the gene-name suffix (so `anti-pdr-1` is dropped
    but `anti-PDR-1` and `anti-C. elegans` are kept), matching the original
    Caltech behavior at main.py:71-72.
    """
    matches: set = set()

    # We need to wrap the sentence in spaces so the leading/trailing char
    # classes in the regexes can match at the boundaries.
    for raw in sentences:
        sentence = " " + _normalize(raw) + " "

        # 1) anti-GENE patterns
        anti_gene_hits = rules.anti_gene_regex.findall(sentence)
        anti_gene_hits = [
            m for m in anti_gene_hits
            if m[5:].lower() != m[5:]  # gene-name suffix must contain at least one uppercase
        ]
        matches.update(anti_gene_hits)

        # 2) Combination patterns (preparation/raised/etc. + antibody/antisera/etc.)
        sentence_lower = sentence.lower()
        for comb, regex_a_first, regex_b_first in rules.combinations_regex:
            if regex_a_first.match(sentence_lower) or regex_b_first.match(sentence_lower):
                matches.add(comb[0] + " " + comb[1])

        # 3) Additional keyword regex (MH46, SP56, a-SP56)
        matches.update(rules.additional_keywords_regex.findall(sentence))

    return matches
```

Key implementation notes (none of these are placeholders — they're decisions):

- The leading/trailing character classes in the original regexes (`[\s\(\[…]`) require a delimiter on both sides. Caltech could rely on its tokenizer; we wrap each sentence in spaces to make the boundary requirement uniform.
- The case-sensitivity filter is verbatim: `m[5:].lower() != m[5:]` — `m[:5]` is `"anti-"`, so `m[5:]` is the gene name; if its lowercase form differs from itself, it had at least one capital.
- En-dash (`–`) and hyphen-bullet (`‐`) are normalized to plain hyphen, matching Caltech's `main.py:66-67`.

- [ ] **Step 3.5: Run tests — expect PASS**

Run: `python -m pytest tests/agr_document_classifier/test_antibody_rules.py -v`
Expected: all 12 tests PASS.

If anything fails, read the failure message carefully — typically it's a regex boundary issue (the wrap-in-spaces step) or the case-sensitivity filter.

- [ ] **Step 3.6: Run flake8 to ensure PEP8 compliance**

Run: `python -m flake8 agr_document_classifier/antibody_rules.py tests/agr_document_classifier/test_antibody_rules.py`
Expected: no output (clean).

- [ ] **Step 3.7: Commit**

```bash
git add agr_document_classifier/antibody_rules.py \
        tests/agr_document_classifier/__init__.py \
        tests/agr_document_classifier/test_antibody_rules.py
git commit -m "feat(SCRUM-5601): add antibody string-matching rule engine

Pure-function port of the Caltech WB antibody pipeline rules:
- anti-GENE regex over curated WB genes plus C. elegans
- combinations of {preparation/raised/...} x {antibody/antiserum/...}
- additional keywords (MH46, SP56, a-SP56)
- excludes PDI; appends MSP to anti-GENE alternatives

Includes the case-sensitivity filter (gene-name suffix must contain at
least one uppercase character) and en-dash normalization.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Create the pipeline entry point

**Files:**
- Create: `agr_document_classifier/agr_antibody_string_matching_classifier.py`
- Create: `tests/agr_document_classifier/test_antibody_classifier_pipeline.py`

- [ ] **Step 4.1: Write failing pipeline tests with extensive mocking**

Create `tests/agr_document_classifier/test_antibody_classifier_pipeline.py`:

```python
"""Integration tests for the antibody string-matching classifier pipeline.

Mocks all ABC API calls and the TEI loader. Exercises the per-job logic
end-to-end: TEI text -> rule matches -> TET POST payload + workflow
status calls.
"""

from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def patched_pipeline():
    """Patch every external dependency the pipeline talks to.

    Yields a dict of mocks keyed by short name so individual tests can
    assert on call args.
    """
    with patch("agr_document_classifier.agr_antibody_string_matching_classifier"
               ".get_all_curated_entities") as m_genes, \
         patch("agr_document_classifier.agr_antibody_string_matching_classifier"
               ".get_tet_source_id", return_value=42) as m_source, \
         patch("agr_document_classifier.agr_antibody_string_matching_classifier"
               ".download_tei_files_for_references") as m_download, \
         patch("agr_document_classifier.agr_antibody_string_matching_classifier"
               ".send_classification_tag_to_abc", return_value=True) as m_send, \
         patch("agr_document_classifier.agr_antibody_string_matching_classifier"
               ".set_job_started", return_value=True) as m_started, \
         patch("agr_document_classifier.agr_antibody_string_matching_classifier"
               ".set_job_success", return_value=True) as m_success, \
         patch("agr_document_classifier.agr_antibody_string_matching_classifier"
               ".set_job_failure", return_value=True) as m_failure, \
         patch("agr_document_classifier.agr_antibody_string_matching_classifier"
               ".AllianceTEI") as m_tei_cls, \
         patch("agr_document_classifier.agr_antibody_string_matching_classifier"
               ".os.listdir", return_value=["AGRKB_101000000000001.tei"]), \
         patch("agr_document_classifier.agr_antibody_string_matching_classifier"
               ".os.remove"), \
         patch("agr_document_classifier.agr_antibody_string_matching_classifier"
               ".os.makedirs"):
        # default: WB-ish gene list
        m_genes.return_value = (["pdr-1", "unc-54"], {}, {})
        yield {
            "genes": m_genes,
            "source": m_source,
            "download": m_download,
            "send": m_send,
            "started": m_started,
            "success": m_success,
            "failure": m_failure,
            "tei_cls": m_tei_cls,
        }


def _job(curie="AGRKB:101000000000001"):
    return {"reference_curie": curie, "reference_workflow_tag_id": 999, "mod_id": 3}


def _set_tei_text(mocks, sentences):
    inst = mocks["tei_cls"].return_value
    inst.load_from_file = MagicMock()
    inst.get_sentences = MagicMock(return_value=sentences)


def test_positive_paper_emits_tet_with_sorted_note(patched_pipeline):
    from agr_document_classifier import agr_antibody_string_matching_classifier as pipe
    _set_tei_text(patched_pipeline, [
        "The antibody was raised against UNC-54.",
        "We also used anti-PDR-1 in some experiments.",
    ])
    pipe.process_antibody_jobs(
        mod_id=3, topic="ATP:0000096",
        jobs=[_job()],
    )

    assert patched_pipeline["send"].call_count == 1
    kwargs = patched_pipeline["send"].call_args.kwargs
    assert kwargs["reference_curie"] == "AGRKB:101000000000001"
    assert kwargs["topic"] == "ATP:0000096"
    assert kwargs["species"] == "NCBITaxon:6239"
    assert kwargs["negated"] is False
    assert kwargs["data_novelty"] == "ATP:0000335"
    assert kwargs["confidence_score"] is None
    assert kwargs["confidence_level"] is None
    # note is sorted, comma-separated
    assert kwargs["note"] == "anti-PDR-1, raised antibody"

    patched_pipeline["started"].assert_called_once()
    patched_pipeline["success"].assert_called_once()
    patched_pipeline["failure"].assert_not_called()


def test_negative_paper_emits_negated_tet_without_note(patched_pipeline):
    from agr_document_classifier import agr_antibody_string_matching_classifier as pipe
    _set_tei_text(patched_pipeline, [
        "We bought a commercial reagent from Sigma.",
        "The animals were maintained at 20 degrees.",
    ])
    pipe.process_antibody_jobs(mod_id=3, topic="ATP:0000096", jobs=[_job()])

    kwargs = patched_pipeline["send"].call_args.kwargs
    assert kwargs["negated"] is True
    assert kwargs["note"] is None
    patched_pipeline["success"].assert_called_once()


def test_tei_parse_failure_marks_job_failed(patched_pipeline):
    from agr_document_classifier import agr_antibody_string_matching_classifier as pipe
    patched_pipeline["tei_cls"].return_value.load_from_file.side_effect = ValueError("bad TEI")
    pipe.process_antibody_jobs(mod_id=3, topic="ATP:0000096", jobs=[_job()])

    patched_pipeline["failure"].assert_called_once()
    patched_pipeline["send"].assert_not_called()
    patched_pipeline["success"].assert_not_called()


def test_empty_curated_gene_list_aborts(patched_pipeline):
    from agr_document_classifier import agr_antibody_string_matching_classifier as pipe
    patched_pipeline["genes"].return_value = ([], {}, {})
    with pytest.raises(RuntimeError, match="curated gene list"):
        pipe.process_antibody_jobs(mod_id=3, topic="ATP:0000096", jobs=[_job()])
    patched_pipeline["send"].assert_not_called()
```

- [ ] **Step 4.2: Run tests — expect FAIL (pipeline module doesn't exist)**

Run: `python -m pytest tests/agr_document_classifier/test_antibody_classifier_pipeline.py -v`
Expected: collection error or `ModuleNotFoundError: No module named 'agr_document_classifier.agr_antibody_string_matching_classifier'`.

- [ ] **Step 4.3: Implement the pipeline**

Create `agr_document_classifier/agr_antibody_string_matching_classifier.py`:

```python
"""WB antibody string-matching topic classifier (SCRUM-5601).

Pulls jobs from ABC, downloads TEIs, applies the rule engine in
`antibody_rules.py`, and POSTs TopicEntityTags with positive/negative +
matched-span notes back to ABC.

Mirrors the structure of agr_document_classifier_classify.py without the
ML-model download/predict path.
"""

import argparse
import copy
import logging
import os
import os.path
import sys
import traceback
from argparse import Namespace

from agr_document_classifier.antibody_rules import build_regexes, match_antibody_spans
from utils.abc_utils import (
    download_tei_files_for_references,
    get_cached_mod_abbreviation_from_id,
    get_cached_mod_id_from_abbreviation,
    get_tet_source_id,
    load_all_jobs,
    send_classification_tag_to_abc,
    set_blue_api_base_url,
    set_job_failure,
    set_job_started,
    set_job_success,
)
from utils.ateam_utils import get_all_curated_entities
from utils.tei_utils import AllianceTEI

from agr_literature_service.lit_processing.utils.report_utils import send_report


logger = logging.getLogger(__name__)


# ---- constants (per design spec §3) -----------------------------------
ANTIBODY_TOPIC = "ATP:0000096"
SOURCE_METHOD = "abc_string_matching_antibody"
SOURCE_DESCRIPTION = (
    "Alliance pipeline that identifies relevant words and/or phrases in "
    "C. elegans references to identify references describing production "
    "and/or use of antibodies."
)
JOB_LABEL = "antibody_string_matching_job"
WB_SPECIES = "NCBITaxon:6239"
DATA_NOVELTY = "ATP:0000335"  # parent term — D2

# data dir conventions match the existing classifier
DATA_DIR = "/data/agr_document_classifier"
TO_CLASSIFY_DIR = f"{DATA_DIR}/to_classify_antibody"
STOP_FILE = f"{DATA_DIR}/stop_antibody_classifier"


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
        force=True,
    )


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(description="WB antibody string-matching topic classifier.")
    parser.add_argument("-l", "--log_level", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO')
    parser.add_argument("-f", "--reference_curie", type=str, required=False,
                        help="Run only for these comma-separated reference curies.")
    parser.add_argument("-m", "--mod_abbreviation", type=str, required=False,
                        help="Run only for this MOD (e.g. WB).")
    parser.add_argument("-t", "--topic", type=str, required=False,
                        help="Run only for this topic ATP id.")
    parser.add_argument("-s", "--stage", action="store_true",
                        help="Hit the stage ABC API.")
    parser.add_argument("--test_mode", action="store_true",
                        help="Run on the supplied references without writing TETs or "
                             "transitioning workflow status.")
    return parser.parse_args()


def _prepare_dir() -> None:
    os.makedirs(TO_CLASSIFY_DIR, exist_ok=True)
    for f in os.listdir(TO_CLASSIFY_DIR):
        try:
            os.remove(os.path.join(TO_CLASSIFY_DIR, f))
        except FileNotFoundError:
            pass


def _curie_from_filename(file_path: str) -> str:
    """`AGRKB_101000000000001.tei` -> `AGRKB:101000000000001`."""
    base = os.path.basename(file_path)
    name = base.split(".")[0]
    return name.replace("_", ":")


def process_antibody_jobs(mod_id, topic: str, jobs: list, *, test_mode: bool = False) -> None:
    """Process all jobs for a single (mod_id, topic). Used by both
    classify_mode (cron) and direct_classify_mode (--test_mode --reference_curie).
    """
    mod_abbr = get_cached_mod_abbreviation_from_id(mod_id) if isinstance(mod_id, int) else mod_id
    if mod_abbr != "WB":
        logger.info(f"Skipping non-WB jobs: mod={mod_abbr}")
        return

    # Build regexes once for the whole batch
    curated_genes, _, _ = get_all_curated_entities(
        mod_abbreviation="WB", entity_type_str="gene")
    if not curated_genes:
        msg = "Empty curated gene list returned from get_all_curated_entities — aborting"
        logger.error(msg)
        raise RuntimeError(msg)

    rules = build_regexes(curated_gene_names=curated_genes)

    tet_source_id = get_tet_source_id(
        mod_abbreviation="WB",
        source_method=SOURCE_METHOD,
        source_description=SOURCE_DESCRIPTION,
    )

    batch_size = int(os.environ.get("CLASSIFICATION_BATCH_SIZE", 1000))
    pending = copy.deepcopy(jobs)
    while pending:
        if os.path.isfile(STOP_FILE):
            logger.info("stop file present — exiting between batches")
            return
        batch = pending[:batch_size]
        pending = pending[batch_size:]
        _process_batch(batch, rules, tet_source_id, topic, test_mode=test_mode)


def _process_batch(batch: list, rules, tet_source_id: int, topic: str,
                   *, test_mode: bool) -> None:
    curie_to_job = {job["reference_curie"]: job for job in batch}
    _prepare_dir()
    download_tei_files_for_references(list(curie_to_job.keys()), TO_CLASSIFY_DIR, "WB")

    for fname in os.listdir(TO_CLASSIFY_DIR):
        path = os.path.join(TO_CLASSIFY_DIR, fname)
        curie = _curie_from_filename(path)
        job = curie_to_job.get(curie)
        if job is None:
            logger.warning(f"TEI {fname} has no matching job; skipping")
            continue

        try:
            tei = AllianceTEI()
            tei.load_from_file(path)
            sentences = tei.get_sentences()
        except Exception as exc:
            logger.error(f"TEI parse failed for {curie}: {exc}")
            if not test_mode:
                set_job_started(job)
                set_job_failure(job)
            continue

        matches = match_antibody_spans(sentences, rules)
        note = ", ".join(sorted(matches)) if matches else None
        negated = len(matches) == 0

        if test_mode:
            logger.info(
                f"[test_mode] {curie}: matches={sorted(matches)} "
                f"negated={negated} note={note}"
            )
            continue

        set_job_started(job)
        ok = send_classification_tag_to_abc(
            reference_curie=curie,
            species=WB_SPECIES,
            topic=topic,
            negated=negated,
            data_novelty=DATA_NOVELTY,
            confidence_score=None,
            confidence_level=None,
            tet_source_id=tet_source_id,
            ml_model_id=None,
            note=note,
        )
        if ok:
            set_job_success(job)
        else:
            set_job_failure(job)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def classify_mode(args: Namespace) -> None:
    logger.info("Antibody string-matching classification started.")
    mod_topic_jobs = load_all_jobs(JOB_LABEL, args)
    failed = []
    for (mod_id, topic), jobs in mod_topic_jobs.items():
        try:
            process_antibody_jobs(mod_id, topic, jobs)
        except Exception as exc:
            logger.error(f"Error processing antibody batch (mod={mod_id} topic={topic}): {exc}")
            failed.append({
                "mod_id": mod_id,
                "topic": topic,
                "exception": str(exc),
                "trace": "<br>".join(traceback.format_tb(exc.__traceback__)),
            })

    if failed:
        msg = "<h>Antibody string-matching pipeline — failed batches:</h><br><br>\n"
        for fp in failed:
            msg += (f"mod_id: {fp['mod_id']}, topic: {fp['topic']}<br>"
                    f"Exception: {fp['exception']}<br>"
                    f"Stacktrace: {fp['trace']}<br><br>\n")
        send_report("Antibody string-matching classification failures", msg)
        sys.exit(-1)


def direct_classify_mode(args: Namespace) -> None:
    if not args.reference_curie or not args.mod_abbreviation:
        logger.error("--test_mode requires --reference_curie and --mod_abbreviation")
        sys.exit(2)

    curies = [c.strip() for c in args.reference_curie.split(",") if c.strip()]
    jobs = [{"reference_curie": c, "reference_workflow_tag_id": None,
             "mod_id": get_cached_mod_id_from_abbreviation(args.mod_abbreviation)}
            for c in curies]
    process_antibody_jobs(
        mod_id=get_cached_mod_id_from_abbreviation(args.mod_abbreviation),
        topic=args.topic or ANTIBODY_TOPIC,
        jobs=jobs,
        test_mode=True,
    )


def main() -> None:
    args = parse_arguments()
    configure_logging(args.log_level)
    if args.stage:
        set_blue_api_base_url("https://stage-literature-rest.alliancegenome.org")
        os.environ['ABC_API_SERVER'] = "https://stage-literature-rest.alliancegenome.org"
        os.environ["ON_PRODUCTION"] = "no"
    else:
        os.environ.setdefault("ON_PRODUCTION", "yes")

    if args.test_mode:
        direct_classify_mode(args)
    else:
        classify_mode(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.4: Run pipeline tests — expect PASS**

Run: `python -m pytest tests/agr_document_classifier/test_antibody_classifier_pipeline.py -v`
Expected: all 4 tests PASS.

If tests fail with `ImportError` on `agr_literature_service`, that import is only used at module-level in `send_report`; mock that too in conftest if needed (the tests above don't trigger `send_report`, but the import-time failure would prevent collection — see step 4.5).

- [ ] **Step 4.5: If `agr_literature_service` is not installed in the test env, lazy-import `send_report`**

If step 4.4 fails at import time with `ModuleNotFoundError: No module named 'agr_literature_service'`, refactor the import in `agr_antibody_string_matching_classifier.py`:

```python
# Replace the top-level import with a lazy local import:
def _send_report(*args, **kwargs):
    from agr_literature_service.lit_processing.utils.report_utils import send_report
    return send_report(*args, **kwargs)
```

And use `_send_report(...)` inside `classify_mode`. Re-run tests.

- [ ] **Step 4.6: Run flake8**

Run: `python -m flake8 agr_document_classifier/agr_antibody_string_matching_classifier.py tests/agr_document_classifier/test_antibody_classifier_pipeline.py`
Expected: no output.

- [ ] **Step 4.7: Commit**

```bash
git add agr_document_classifier/agr_antibody_string_matching_classifier.py \
        tests/agr_document_classifier/test_antibody_classifier_pipeline.py
git commit -m "feat(SCRUM-5601): add WB antibody string-matching pipeline entry point

Pulls jobs from ABC via load_all_jobs, downloads TEI files, applies the
antibody_rules engine, and POSTs TopicEntityTags with positive/negative
classification plus the comma-joined matched-span note. Mirrors the
existing classifier's structure but skips the ML-model download/predict
path. Includes --test_mode for ad-hoc curie verification.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add Makefile target

**Files:**
- Modify: `Makefile`

- [ ] **Step 5.1: Append the target**

Append to `Makefile`:

```makefile
classify_antibody:
	docker-compose --env-file ${ENV_FILE} run agr_automated_information_extraction \
	    python agr_document_classifier/agr_antibody_string_matching_classifier.py
```

- [ ] **Step 5.2: Verify the target is recognized**

Run: `make -n classify_antibody` (dry-run)
Expected: prints the docker-compose command without errors.

- [ ] **Step 5.3: Commit**

```bash
git add Makefile
git commit -m "chore(SCRUM-5601): add classify_antibody Make target

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: agr_literature_service — transitions oneoff

**Switch to the agr_literature_service repo for tasks 6 and 7.**

```bash
cd /home/valerio/workspace/agr/agr_literature_service
git status   # confirm we're on SCRUM-5601 with a clean tree
```

**Files:**
- Create: `agr_literature_service/lit_processing/oneoff_scripts/SCRUM-5601_add_antibody_str_match_transitions.py`

This script writes to a Postgres DB. It cannot be unit-tested without a fixture DB, so verification is via dry-run logging on stage.

- [ ] **Step 6.1: Write the script**

Create the file with:

```python
"""SCRUM-5601: Register the workflow_transition rows that drive the WB
antibody string-matching topic classifier.

What this script does (idempotent — safe to re-run):

  (a) INSERTs four new transitions for WB:
       - reference classification needed (ATP:0000166) -> ATP:0000366
         (condition: 'antibody_string_matching_job') makes ATP:0000366
         poll-able by load_all_jobs("antibody_string_matching_job")
       - ATP:0000366 -> ATP:0000365 (on_start)
       - ATP:0000365 -> ATP:0000363 (on_success)
       - ATP:0000365 -> ATP:0000364 (on_failed)

  (b) UPDATEs the two existing WB 'text conversion needed/in progress
      -> file converted to text (on_success)' rows by appending
       'proceed_on_value::reference_type::Experimental::ATP:0000366' to
      their actions arrays — so newly text-converted WB Experimental
      references automatically receive ATP:0000366 alongside the other
      classification-needed tags.

ATP curies (confirmed):
  ATP:0000162 = text conversion needed
  ATP:0000198 = text conversion in progress
  ATP:0000163 = file converted to text
  ATP:0000166 = reference classification needed
  ATP:0000363 = antibody string matching classification complete
  ATP:0000364 = antibody string matching classification failed
  ATP:0000365 = antibody string matching classification in progress
  ATP:0000366 = antibody string matching classification needed
"""

import logging
from os import path

from sqlalchemy import text

from agr_literature_service.lit_processing.utils.sqlalchemy_utils import \
    create_postgres_session
from agr_literature_service.api.user import set_global_user_id

logging.basicConfig(format='%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


WB = "WB"

REF_CLASSIFICATION_NEEDED = "ATP:0000166"
ANTIBODY_NEEDED = "ATP:0000366"
ANTIBODY_IN_PROGRESS = "ATP:0000365"
ANTIBODY_COMPLETE = "ATP:0000363"
ANTIBODY_FAILED = "ATP:0000364"

TEXT_CONV_NEEDED = "ATP:0000162"
TEXT_CONV_IN_PROGRESS = "ATP:0000198"
FILE_CONVERTED = "ATP:0000163"

NEW_ACTION = (
    f"proceed_on_value::reference_type::Experimental::{ANTIBODY_NEEDED}"
)


# (transition_from, transition_to, condition, actions)
NEW_TRANSITIONS = [
    (REF_CLASSIFICATION_NEEDED, ANTIBODY_NEEDED, "antibody_string_matching_job", []),
    (ANTIBODY_NEEDED, ANTIBODY_IN_PROGRESS, "on_start",
        ["sub_task_in_progress::reference classification"]),
    (ANTIBODY_IN_PROGRESS, ANTIBODY_COMPLETE, "on_success",
        ["sub_task_complete::reference classification"]),
    (ANTIBODY_IN_PROGRESS, ANTIBODY_FAILED, "on_failed",
        ["sub_task_failed::reference classification"]),
]


def insert_new_transitions(db, mod_id):
    inserted = 0
    skipped = 0
    for trans_from, trans_to, condition, actions in NEW_TRANSITIONS:
        existing = db.execute(text("""
            SELECT 1 FROM workflow_transition
            WHERE mod_id = :mod_id
              AND transition_from = :tf
              AND transition_to = :tt
              AND COALESCE(condition, '') = :cond
        """), {"mod_id": mod_id, "tf": trans_from, "tt": trans_to,
               "cond": condition or ""}).first()
        if existing:
            logger.info(f"  [skip] {trans_from} -> {trans_to} "
                        f"(condition='{condition}') already exists")
            skipped += 1
            continue
        db.execute(text("""
            INSERT INTO workflow_transition
                (mod_id, transition_from, transition_to, condition, actions,
                 transition_type, date_created)
            VALUES
                (:mod_id, :tf, :tt, :cond, :actions, 'any', NOW())
        """), {"mod_id": mod_id, "tf": trans_from, "tt": trans_to,
               "cond": condition, "actions": actions})
        logger.info(f"  [insert] {trans_from} -> {trans_to} "
                    f"(condition='{condition}', actions={actions})")
        inserted += 1
    db.commit()
    logger.info(f"transitions: inserted={inserted} skipped={skipped}")


def append_action_to_text_conversion_rows(db, mod_id):
    """Append NEW_ACTION to the WB on_success rows transitioning into
    'file converted to text'. Uses array_append + a NOT-already-in guard
    to stay idempotent.
    """
    sql = text("""
        UPDATE workflow_transition
           SET actions = array_append(COALESCE(actions, ARRAY[]::text[]), :new_action)
         WHERE mod_id = :mod_id
           AND transition_to = :file_converted
           AND transition_from IN (:tc_needed, :tc_in_progress)
           AND condition = 'on_success'
           AND NOT (:new_action = ANY(COALESCE(actions, ARRAY[]::text[])))
    """)
    result = db.execute(sql, {
        "new_action": NEW_ACTION,
        "mod_id": mod_id,
        "file_converted": FILE_CONVERTED,
        "tc_needed": TEXT_CONV_NEEDED,
        "tc_in_progress": TEXT_CONV_IN_PROGRESS,
    })
    db.commit()
    logger.info(f"text-conversion action append: rows updated = {result.rowcount}")


def main():
    db = create_postgres_session(False)
    set_global_user_id(db, path.basename(__file__).replace(".py", ""))

    mod_row = db.execute(text("SELECT mod_id FROM mod WHERE abbreviation = :m"),
                         {"m": WB}).fetchone()
    if not mod_row:
        logger.error(f"mod '{WB}' not found in mod table")
        return
    mod_id = int(mod_row[0])
    logger.info(f"Operating on mod_id={mod_id} ({WB})")

    logger.info("(a) inserting four-state transitions for antibody string matching")
    insert_new_transitions(db, mod_id)

    logger.info("(b) appending action to text-conversion -> file converted to text rows")
    append_action_to_text_conversion_rows(db, mod_id)

    logger.info("done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Lint**

Run: `python -m flake8 agr_literature_service/lit_processing/oneoff_scripts/SCRUM-5601_add_antibody_str_match_transitions.py`
Expected: no output.

- [ ] **Step 6.3: Commit**

```bash
git add agr_literature_service/lit_processing/oneoff_scripts/SCRUM-5601_add_antibody_str_match_transitions.py
git commit -m "feat(SCRUM-5601): oneoff to register antibody string-matching transitions

Adds four workflow_transition rows for the WB antibody-string-matching
state machine and appends the proceed_on_value action to the two existing
WB text-conversion -> file-converted-to-text rows. Idempotent (NOT-EXISTS
guard for inserts, ANY-array guard for the action append).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: agr_literature_service — backfill oneoff

**Files:**
- Create: `agr_literature_service/lit_processing/oneoff_scripts/SCRUM-5601_backfill_antibody_str_match_needed_tag.py`

- [ ] **Step 7.1: Write the script**

Create the file with:

```python
"""SCRUM-5601: backfill ATP:0000366 (antibody string matching classification
needed) for WB in-corpus references the legacy caltech antibody import did
NOT cover.

Skip rules:
  1. Skip references that already have a TET emitted by the caltech import
     (topic_entity_tag_source.source_method = 'string_matching_antibody').
  2. Skip references that already have any tag in the new four-state
     antibody-string-matching workflow process (idempotent re-runs).
"""

import logging
from os import path

from sqlalchemy import bindparam, text

from agr_literature_service.api.models import WorkflowTagModel
from agr_literature_service.api.user import set_global_user_id
from agr_literature_service.lit_processing.utils.sqlalchemy_utils import \
    create_postgres_session


logging.basicConfig(format='%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


WB = "WB"
ANTIBODY_NEEDED = "ATP:0000366"
ANTIBODY_IN_PROGRESS = "ATP:0000365"
ANTIBODY_COMPLETE = "ATP:0000363"
ANTIBODY_FAILED = "ATP:0000364"
PROCESS_TAGS = (ANTIBODY_NEEDED, ANTIBODY_IN_PROGRESS,
                ANTIBODY_COMPLETE, ANTIBODY_FAILED)
LEGACY_CALTECH_SOURCE_METHOD = "string_matching_antibody"

BATCH_COMMIT_SIZE = 250


def backfill():
    db = create_postgres_session(False)
    set_global_user_id(db, path.basename(__file__).replace(".py", ""))

    mod_row = db.execute(text("SELECT mod_id FROM mod WHERE abbreviation = :m"),
                         {"m": WB}).fetchone()
    if not mod_row:
        logger.error(f"mod '{WB}' not found")
        return
    mod_id = int(mod_row[0])

    in_corpus = {r[0] for r in db.execute(text("""
        SELECT reference_id
          FROM mod_corpus_association
         WHERE mod_id = :mod_id
           AND corpus = TRUE
    """), {"mod_id": mod_id})}

    legacy_covered = {r[0] for r in db.execute(text("""
        SELECT DISTINCT tet.reference_id
          FROM topic_entity_tag tet
          JOIN topic_entity_tag_source tets
            ON tet.topic_entity_tag_source_id = tets.topic_entity_tag_source_id
         WHERE tets.source_method = :sm
    """), {"sm": LEGACY_CALTECH_SOURCE_METHOD})}

    in_new_process = {r[0] for r in db.execute(text("""
        SELECT reference_id FROM workflow_tag
         WHERE mod_id = :mod_id AND workflow_tag_id IN :tags
    """).bindparams(bindparam("tags", expanding=True)),
        {"mod_id": mod_id, "tags": list(PROCESS_TAGS)})}

    missing = in_corpus - legacy_covered - in_new_process
    logger.info(
        f"WB in-corpus: {len(in_corpus)} | "
        f"legacy-caltech-covered: {len(legacy_covered & in_corpus)} | "
        f"already in new process: {len(in_new_process & in_corpus)} | "
        f"to backfill: {len(missing)}"
    )

    inserted = 0
    for ref_id in missing:
        db.add(WorkflowTagModel(
            reference_id=ref_id,
            mod_id=mod_id,
            workflow_tag_id=ANTIBODY_NEEDED,
        ))
        inserted += 1
        if inserted % BATCH_COMMIT_SIZE == 0:
            db.commit()
            logger.info(f"  committed {inserted} so far")
    db.commit()
    logger.info(f"inserted {inserted} ATP:0000366 tags")


if __name__ == "__main__":
    backfill()
```

- [ ] **Step 7.2: Lint**

Run: `python -m flake8 agr_literature_service/lit_processing/oneoff_scripts/SCRUM-5601_backfill_antibody_str_match_needed_tag.py`
Expected: no output.

- [ ] **Step 7.3: Commit**

```bash
git add agr_literature_service/lit_processing/oneoff_scripts/SCRUM-5601_backfill_antibody_str_match_needed_tag.py
git commit -m "feat(SCRUM-5601): oneoff backfill for antibody string-matching needed tag

Inserts ATP:0000366 for WB in-corpus references that the legacy caltech
import did NOT cover (no TET with source_method=string_matching_antibody).
Skips references that already have any tag in the new four-state process
so re-runs are idempotent. Commits every 250 inserts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Stage smoke test

This task runs against the stage environment and is best done interactively after the PRs are merged and deployed. It is included here so the engineer doesn't forget the verification step.

- [ ] **Step 8.1: Run the transitions oneoff against stage**

```bash
cd /home/valerio/workspace/agr/agr_literature_service
# (with stage DB credentials in PSQL_USERNAME/PASSWORD/HOST/PORT/DATABASE env)
python agr_literature_service/lit_processing/oneoff_scripts/SCRUM-5601_add_antibody_str_match_transitions.py
```

Expected output: `transitions: inserted=4 skipped=0` and `text-conversion action append: rows updated = 2`. Re-run: `inserted=0 skipped=4` and `rows updated = 0` (idempotent).

- [ ] **Step 8.2: Spot-check a fresh WB Experimental reference**

Pick (or create) a WB reference whose `reference_type = "Experimental"`. Walk it through corpus add → file upload → text conversion → file converted to text. Expected: `workflow_tag` table has a row with `(reference_id=<ref>, mod_id=<WB>, workflow_tag_id='ATP:0000366')`.

- [ ] **Step 8.3: Run the backfill oneoff against stage**

```bash
python agr_literature_service/lit_processing/oneoff_scripts/SCRUM-5601_backfill_antibody_str_match_needed_tag.py
```

Expected: log line printing the counts, then `inserted N ATP:0000366 tags` (N matches the "to backfill" count).

- [ ] **Step 8.4: Run the antibody pipeline in --test_mode against a small set**

```bash
cd /home/valerio/workspace/agr/agr_automated_information_extraction
python agr_document_classifier/agr_antibody_string_matching_classifier.py \
    --stage --test_mode \
    --mod_abbreviation WB \
    --topic ATP:0000096 \
    --reference_curie AGRKB:101000000XXXXXXX,AGRKB:101000000YYYYYYY \
    --log_level INFO
```

(Replace X/Y curies with two known WB papers — one expected positive, one expected negative. Daniela can suggest examples.)

Expected: log lines `[test_mode] AGRKB:...: matches=[...] negated=False note='...'` for the positive paper and `negated=True note=None` for the negative one. No TETs are POSTed in test mode.

- [ ] **Step 8.5: Run the antibody pipeline live on stage for a small batch**

After verifying in --test_mode, run without --test_mode:

```bash
python agr_document_classifier/agr_antibody_string_matching_classifier.py \
    --stage \
    --mod_abbreviation WB \
    --topic ATP:0000096 \
    --log_level INFO
```

Limit by setting `CLASSIFICATION_BATCH_SIZE=10` and creating `/data/agr_document_classifier/stop_antibody_classifier` after the first batch. Verify TETs in the stage UI for the processed papers.

- [ ] **Step 8.6: Production rollout (only after Daniela signs off on stage)**

Repeat 8.1, 8.3, then 8.5 (no batch limit) against production. Schedule the pipeline via cron alongside the existing classifier.

---

## Self-review notes

- **Spec coverage:** every section in the spec maps to a task — abc_utils.note + eco_map (T1, T2), rule engine (T3), pipeline (T4), Makefile (T5), transitions oneoff (T6), backfill oneoff (T7), stage smoke (T8).
- **Type consistency:** `build_regexes` returns `AntibodyRules` (NamedTuple) used by `match_antibody_spans` — name and field order match across the rule module and the pipeline. `process_antibody_jobs(mod_id, topic, jobs, *, test_mode=False)` signature matches the call sites in `classify_mode` and `direct_classify_mode`.
- **No placeholders:** all code blocks are complete; all commands are runnable.

## Known caveats the engineer should be aware of

1. The `agr_literature_service` repo is a separate local clone at `/home/valerio/workspace/agr/agr_literature_service`. Tasks 6-7 commit there, on its own `SCRUM-5601` branch (already created). Do **not** mix those commits onto the `agr_automated_information_extraction` branch.
2. If `python -m pytest tests/agr_document_classifier/test_antibody_classifier_pipeline.py` fails at collection due to `agr_literature_service` not being import-resolvable in the test env, follow Step 4.5 to lazy-import `send_report`.
3. Stage smoke (T8) requires DB credentials and access to the stage ABC environment — the engineer running this plan needs them or needs to coordinate with someone who does.
