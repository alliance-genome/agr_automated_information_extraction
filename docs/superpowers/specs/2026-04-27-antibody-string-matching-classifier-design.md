# WB Antibody String-Matching Topic Classifier — Design

- **Jira:** [SCRUM-5601](https://agr-jira.atlassian.net/browse/SCRUM-5601)
- **Branches:** `SCRUM-5601` in `agr_automated_information_extraction` and in `agr_literature_service`
- **Status:** Draft, awaiting design approval before plan file is written
- **Author:** Valerio Arnaboldi (with Claude)
- **Date:** 2026-04-27

## 1. Background

The WB curation team currently runs a string-matching antibody-detection pipeline
at Caltech (`/home/valerio/workspace/caltech/entity-extraction-antibody/main.py`).
It reads papers from the WormBase Postgres DB via `wbtools`, runs three rule sets
over the full text, stores the matched spans in the WB DB, and emails a weekly TSV
digest. SCRUM-5601 asks us to migrate this pipeline to the Alliance Biocuration
Collective (ABC) so that:

1. Papers are pulled from ABC instead of the WB DB.
2. Results land as ABC TopicEntityTags (TET) rather than rows in the WB DB.
3. Workflow status is tracked via the new ATP terms registered for this process.
4. The matched-span text is preserved on the TET as a `note`.
5. No more weekly email; curators read results from the ABC UI.

Daniela has confirmed (Jira comment, 2026-04-27) that the Caltech pipeline rules
are still current and need no changes.

## 2. Scope

This work spans **two repositories** on a shared `SCRUM-5601` branch:

### 2.1 `agr_automated_information_extraction`
- New string-matching topic classifier under `agr_document_classifier/`.
- Small modification to `utils/abc_utils.py` to support a TET `note` field and
  register the new TET source method.

### 2.2 `agr_literature_service`
- Auto-create the *antibody string matching classification needed* workflow tag
  (ATP:0000366) on any reference newly added to the WB corpus.
- One-off backfill script that adds ATP:0000366 to all existing WB in-corpus
  references that don't have any tag in the parent workflow process yet.

Out of scope: registering the new ATP workflow process / status tags in the
Alliance ontology (assumed already done by the curation/ATP team — Ceri's
comment lists the assigned IDs), and the eventual NN-based antibody classifier
(SCRUM-4688 covers training data; will run alongside this string matcher).

## 3. Constants and identifiers

| Item | Value | Notes |
|---|---|---|
| Topic ATP (TET payload) | `ATP:0000096` | "antibody" topic |
| WF process root | (parent of the four below — TBD by curation/ATP team) | |
| WF "needed" | `ATP:0000366` | antibody string matching classification needed |
| WF "in progress" | `ATP:0000365` | |
| WF "complete" | `ATP:0000363` | |
| WF "failed" | `ATP:0000364` | |
| Source `source_method` | `abc_antibody_string_matching_classifier` | Specific to this pipeline |
| Source `source_evidence_assertion` | `ECO:0008021` | string matching |
| MOD | `WB` only | Sources are MOD-specific |
| Species | `NCBITaxon:6239` | from `MOD_TAXON_MAPPING` in `ateam_utils.py` |
| Job condition / `job_str` | `antibody_string_matching_job` | Deliberately omits the `classification_job` substring to avoid collision with the ML doc classifier's `load_all_jobs("classification_job", …)` poll. The substring filter is exclusive to this pipeline. |

## 4. Architecture — `agr_automated_information_extraction`

### 4.1 Component breakdown

```
agr_document_classifier/
├── antibody_rules.py                              [NEW]
│       Pure rule engine; no I/O; trivially unit-testable.
├── agr_antibody_string_matching_classifier.py     [NEW]
│       Pipeline entry point; mirrors agr_document_classifier_classify.py
│       structure but skips the ML-model download/predict path.
└── (existing files untouched)

utils/
└── abc_utils.py                                   [MODIFIED]
        + note: Optional[str] = None on send_classification_tag_to_abc
        + new entry in get_tet_source_id eco_map

tests/agr_document_classifier/
├── test_antibody_rules.py                         [NEW]
└── test_antibody_classifier_pipeline.py           [NEW]

Makefile                                            [MODIFIED, optional]
        + classify_antibody target
```

### 4.2 `antibody_rules.py`

Pure module, no ABC imports.

```python
EXCLUDE_GENES        = ['PDI']
ADDITIONAL_ANTI_KEYWORDS = ['MSP']            # added to the curated gene list before regex build
ADDITIONAL_KEYWORDS  = ['MH46', 'SP56', 'a-SP56']

COMBINATION_1 = ["preparation", "prepared", "prepare", "production",
                 "purification", "generation", "generate", "generated",
                 "produce", "produced", "purify", "purified", "raised"]
COMBINATION_2 = ["antiserum", "antibody", "antibodies", "antisera"]


def build_regexes(curated_gene_names: list[str]):
    """Return a tuple (anti_gene_regex, combinations_regex_pairs, additional_regex).
    Compiled once per pipeline run, reused per paper.
    """

def match_antibody_spans(sentences: list[str], regexes) -> set[str]:
    """Apply rules to a normalized list of sentences. Returns the union of matches."""
```

The three rule sets are ported verbatim from `caltech/.../main.py:50-82`,
including the dash-normalization (`–`/`‐` → `-`) and the case-sensitivity
filter on `anti-GENE` matches that requires capitalization in the suffix.

### 4.3 `agr_antibody_string_matching_classifier.py`

Mirrors the structure of `agr_document_classifier_classify.py` so curators
already familiar with the existing pipeline find it immediately legible:

```python
def parse_arguments():            # same args as agr_document_classifier_classify.py:
                                  #   --reference_curie, --mod_abbreviation,
                                  #   --topic, --stage, --test_mode, --log_level
def classify_mode(args):          # batches via load_all_jobs(JOB_LABEL, args)
def direct_classify_mode(args):   # ad-hoc single-curie runs (test mode)
def process_antibody_jobs(mod_id, topic, jobs, regexes, tet_source_id, species):
                                  # iterate batches, call process_job_batch
def process_job_batch(...):       # download TEIs, run extract_antibody_matches per file
def extract_antibody_matches(tei_obj, regexes) -> set[str]:
                                  # tei_obj.get_sentences() → match_antibody_spans
def send_results(...)             # call send_classification_tag_to_abc
def main()
```

Key differences from the ML classifier:
- No `load_embedding_model`, no `download_abc_model`, no `get_model_data` calls.
- Curated WB gene list fetched once via
  `get_all_curated_entities(mod_abbreviation="WB", entity_type_str="gene")`,
  used to construct the regexes (with `EXCLUDE_GENES` removed and
  `ADDITIONAL_ANTI_KEYWORDS` appended, exactly matching Caltech behavior).
- `ml_model_id=None`, `confidence_score`/`confidence_level`/`data_novelty` —
  see open questions 1 and 2.
- Topic is fixed (`ATP:0000096`); we still group by `(mod_id, topic)` from
  `load_all_jobs` for symmetry with the existing pipeline, but expect a single
  group in practice.

### 4.4 Data flow

```
ABC: WB reference enters corpus
  → mod_corpus_association_crud.create() (literature_service change, §5.1)
  → WorkflowTagModel(reference_id, mod_id=WB, workflow_tag_id=ATP:0000366) inserted

Pipeline run (cron / on-demand):
  load_all_jobs("antibody_string_matching_job", args)
  → [(WB_mod_id, ATP:0000096): [{reference_curie, reference_workflow_tag_id, …}, …]]

For each batch (size: CLASSIFICATION_BATCH_SIZE, default 1000):
  download_tei_files_for_references(curies, /data/.../to_classify, "WB")

  For each TEI file:
    set_job_started(job)            # transitions WF tag to ATP:0000365
    AllianceTEI().load_from_file()
    sentences = tei.get_sentences()
    matches   = match_antibody_spans(sentences, regexes)

    note = ", ".join(sorted(matches)) if matches else None
    negated = (len(matches) == 0)

    send_classification_tag_to_abc(
        reference_curie=curie,
        species="NCBITaxon:6239",
        topic="ATP:0000096",
        negated=negated,
        note=note,
        confidence_score=<see Q1>, confidence_level=<see Q1>,
        data_novelty=<see Q2>,
        tet_source_id=tet_source_id,
        ml_model_id=None,
    )

    set_job_success(job)            # transitions WF tag to ATP:0000363
                                    # (or set_job_failure → ATP:0000364 on TEI parse failure)
    os.remove(tei_file)
```

The ABC backend is responsible for moving the workflow tag through its
ATP states based on `set_job_started/success/failure` calls; the pipeline
does not call `create_workflow_tag` directly. (This already works for the
existing classifier and entity extractor.)

### 4.5 `utils/abc_utils.py` changes

Two surgical edits, both backward-compatible:

**a)** Add `note` parameter to `send_classification_tag_to_abc` (currently at
line 212). Include it in the JSON payload only when not `None`:

```python
def send_classification_tag_to_abc(reference_curie, species, topic, negated,
                                   data_novelty, confidence_score,
                                   confidence_level, tet_source_id,
                                   ml_model_id=None, note=None):   # <-- added
    payload = { …existing fields… }
    if note is not None:
        payload["note"] = note
    tet_data = json.dumps(payload).encode("utf-8")
```

The ABC schema (`TopicEntityTagSchemaCreate.note`,
`agr_literature_service/api/schemas/topic_entity_tag_schemas.py:81`) already
accepts the field. The CRUD layer
(`api/crud/topic_entity_tag_crud.py:759-811`) handles append-on-duplicate
with `" | "` separator — useful if the same paper is reprocessed.

**b)** Add to `eco_map` in `get_tet_source_id` (line 132):

```python
eco_map = {
    "abc_entity_extractor":                  "ECO:0008021",
    "abc_document_classifier":               "ECO:0008004",
    "abc_literature_system":                 "ATP:0000036",
    "abc_antibody_string_matching_classifier": "ECO:0008021",   # <-- added
}
```

The function already POSTs a new source row when none exists for that
`(eco_code, source_method, mod_abbreviation)` triplet, so the source is
provisioned automatically on first run in each environment (dev, stage,
prod).

### 4.6 Error handling

| Condition | Action |
|---|---|
| TEI download failed (no file produced) | log warning, leave job in "in progress" — next run retries |
| TEI parse failure (`AllianceTEI.load_from_file` raises) | `set_job_failure` → ATP:0000364 |
| Empty fulltext / no sentences | treat as "no matches" → negated TET, no note, `set_job_success` (absence of evidence is a valid result) |
| `get_all_curated_entities("WB", "gene")` returns empty | abort the whole run with a loud error via `send_report` (an empty gene list silently corrupts results) |
| TET POST failure | bubble up; the existing 3-attempt retry inside `send_classification_tag_to_abc` handles transient errors. Final failure ⇒ `set_job_failure`. |
| Any uncaught exception in `process_antibody_jobs` | mirrored from `agr_document_classifier_classify.py:classify_mode` — caught at the per-(mod, topic) loop, accumulated in `failed_processes`, sent via `send_report` at end |
| `/data/agr_document_classifier/stop_classifier` exists | honor between batches, same as the ML classifier |

### 4.7 Testing

**Unit:**
- `tests/agr_document_classifier/test_antibody_rules.py` — cover each rule:
  - `anti-PDR-1 antibody` → matches `anti-pdr-1`
  - `anti-PDI antibody` → no match (PDI excluded)
  - `anti-pdr-1` (no capitalization on suffix) → no match (case filter at
    Caltech `main.py:71-72`)
  - `antibody was raised against UNC-54` → matches `raised antibody`
  - `MH46 was used` → matches `MH46`
  - `commercial antibody from Sigma` (no rule trigger) → no match
- `tests/agr_document_classifier/test_antibody_classifier_pipeline.py` —
  mock `download_tei_files_for_references`, `send_classification_tag_to_abc`,
  `set_job_started/success/failure`, `get_all_curated_entities`. Assert:
  - one TET POST per reference
  - correct payload shape (negated when no matches; note populated and
    sorted+`", "`-joined when matches)
  - workflow transitions called in order

**Integration / smoke:**
- `--test_mode --reference_curie AGRKB:... --mod_abbreviation WB --topic ATP:0000096 --stage`
  hitting stage. Verify TET appears in the stage UI with expected note.

## 5. Architecture — `agr_literature_service`

The auto-create trigger plugs into the **existing** "text conversion → file
converted to text" transition for WB Experimental references. That row's
`actions` column already creates `disease classification needed`,
`expression classification needed`, etc. via `proceed_on_value`. We extend
that array to include `ATP:0000366` and register the new four-state
machine.

Critically: the `lit_processing/oneoff_scripts/workflow/data/*.py` files
(`text_conversion.py`, `classification.py`, etc.) are **historical seed
data**, not living source of truth. They were the initial bootstrap when
each workflow process was first defined. New cross-cutting changes follow
the precedent of `6755b395f397-SCRUM-4112.py` — a per-ticket oneoff
script that talks to `workflow_transition` directly. We don't touch the
`data/` files.

That also means **no code change** in `mod_corpus_association_crud.py`.
All the work is in two new oneoff scripts.

### 5.1 Oneoff: register the antibody string-matching workflow transitions

**File:** `lit_processing/oneoff_scripts/SCRUM-5601_add_antibody_str_match_transitions.py`
[NEW]

Idempotent script that does two things on a database:

**a) INSERT four new `workflow_transition` rows for WB**, mirroring the
state-machine the existing classifier processes have:

| transition_from | transition_to | condition | actions | Why |
|---|---|---|---|---|
| `ATP:0000166` (`reference classification needed`) | `ATP:0000366` (`antibody string matching classification needed`) | `antibody_string_matching_job` | none | makes ATP:0000366 poll-able as a job (join target for `load_all_jobs("antibody_string_matching_job")`) |
| `ATP:0000366` (needed) | `ATP:0000365` (in progress) | `on_start` | `["sub_task_in_progress::reference classification"]` | `set_job_started` path |
| `ATP:0000365` (in progress) | `ATP:0000363` (complete) | `on_success` | `["sub_task_complete::reference classification"]` | `set_job_success` path |
| `ATP:0000365` (in progress) | `ATP:0000364` (failed) | `on_failed` | `["sub_task_failed::reference classification"]` | `set_job_failure` path |

The `transition_from` for the first row uses the existing parent
`ATP:0000166` "reference classification needed" tag (the same parent the
other classifier subtasks hang off of in the seed `classification.py:38-44`,
docstring at `classification.py:23`). All ATP curies are hardcoded in a
small dict at the top of the script — same pattern as
`6755b395f397-SCRUM-4112.py:14-19`.

The script uses `INSERT … ON CONFLICT DO NOTHING` (or a pre-`SELECT` check
on `(mod_id, transition_from, transition_to)` if the table doesn't have a
unique constraint) so it's safe to re-run.

**b) UPDATE the two existing WB `text conversion → file converted to text`
rows** to append the new `proceed_on_value` action for ATP:0000366:

```sql
UPDATE workflow_transition
SET actions = array_append(
    actions,
    'proceed_on_value::reference_type::Experimental::ATP:0000366')
WHERE mod_id = (SELECT mod_id FROM mod WHERE abbreviation = 'WB')
  AND transition_to = 'ATP:0000163'      -- "file converted to text"
  AND transition_from IN (
        'ATP:0000162',                   -- "text conversion needed"
        'ATP:0000198')                   -- "text conversion in progress"
  AND condition = 'on_success'
  AND NOT (
      'proceed_on_value::reference_type::Experimental::ATP:0000366' = ANY(actions)
  );
```

ATP curies confirmed: `ATP:0000162` (text conversion needed),
`ATP:0000198` (text conversion in progress), `ATP:0000163` (file converted
to text). The two existing rows targeted by this UPDATE are the WB
"on_success" transitions defined at `text_conversion.py:26-49`.

The `NOT (… = ANY(actions))` guard makes the UPDATE idempotent.

After running this once per environment (dev, stage, prod), every newly-
ingested WB Experimental reference will receive ATP:0000366 automatically
once its text conversion succeeds, alongside the other five classification-
needed tags. No more code changes needed in `mod_corpus_association_crud`,
no recurring data-file maintenance.

### 5.2 Oneoff: backfill ATP:0000366 for existing WB in-corpus references

**File:** `lit_processing/oneoff_scripts/SCRUM-5601_backfill_antibody_str_match_needed_tag.py`
[NEW]

The transitions in §5.1 only fire on *new* text-conversion success events,
so all WB references that already passed text conversion before this
change ships need a one-time backfill. Pattern: `backfill_file_upload_WFT.py`
in the same directory.

Sketch:

```python
"""Backfill ATP:0000366 (antibody string matching classification needed) for all
WB in-corpus references that don't have a tag in the antibody-string-matching
classification workflow process yet.
"""

WB = "WB"
ANTIBODY_STR_NEEDED       = "ATP:0000366"
ANTIBODY_STR_IN_PROGRESS  = "ATP:0000365"
ANTIBODY_STR_COMPLETE     = "ATP:0000363"
ANTIBODY_STR_FAILED       = "ATP:0000364"
PROCESS_TAGS = (ANTIBODY_STR_NEEDED, ANTIBODY_STR_IN_PROGRESS,
                ANTIBODY_STR_COMPLETE, ANTIBODY_STR_FAILED)


def backfill():
    db = create_postgres_session(False)
    set_global_user_id(db, "backfill_antibody_str_match_needed_tag")
    mod_id = db.execute(text("SELECT mod_id FROM mod WHERE abbreviation = :m"),
                        {"m": WB}).scalar()

    # all WB in-corpus references
    in_corpus = set(r[0] for r in db.execute(text("""
        SELECT reference_id FROM mod_corpus_association
        WHERE mod_id = :mod_id AND corpus = TRUE
    """), {"mod_id": mod_id}))

    # references that already have ANY tag in this process
    have_tag = set(r[0] for r in db.execute(text("""
        SELECT reference_id FROM workflow_tag
        WHERE mod_id = :mod_id AND workflow_tag_id IN :tags
    """).bindparams(bindparam("tags", expanding=True)),
        {"mod_id": mod_id, "tags": PROCESS_TAGS}))

    missing = in_corpus - have_tag
    logger.info(f"WB in-corpus references missing antibody process tag: {len(missing)}")

    inserted = 0
    for ref_id in missing:
        db.add(WorkflowTagModel(
            reference_id=ref_id, mod_id=mod_id,
            workflow_tag_id=ANTIBODY_STR_NEEDED))
        inserted += 1
        if inserted % 250 == 0:
            db.commit()
    db.commit()
    logger.info(f"Inserted {inserted} ATP:0000366 tags")


if __name__ == "__main__":
    backfill()
```

The script:
- Touches WB only.
- Skips references that already have *any* tag in the four-state process
  (i.e. don't overwrite an in-progress / complete / failed state from a
  previous run).
- Filters references by publication date if a `--since YYYY-MM-DD` argument
  is provided; otherwise backfills all in-corpus references. The exact
  cutoff to use at rollout is open — see Q4 in §6.
- Commits in batches of 250 to keep transaction sizes reasonable, matching
  the existing `backfill_file_upload_WFT.py` style.

**Test plan:** run on a stage DB snapshot, count tags before/after, sanity
check that no non-WB or non-corpus references were touched.

## 6. Open questions

Only the items below remain undecided; everything else (job condition format,
auto-create mechanism, backfill cutoff) is settled in §3 / §5.

### Q1. `confidence_score` / `confidence_level` for a deterministic classifier
The existing `send_classification_tag_to_abc` requires both. Two reasonable
choices:
- **`1.0` / `"HIGH"`** — accurate (string matching is deterministic; if a
  rule fired, we are 100% sure it fired) and matches `get_confidence_level`
  thresholds (`agr_document_classifier_classify.py:228`). Downside:
  conflates "rule fired" with "biological certainty"; a curator filtering
  "high confidence positives" might over-trust string matches.
- **`None` / `None`** — the schema's `Optional`. Communicates "score not
  applicable to this method." Downside: breaks any UI assumption that
  every TET has a score.

**Recommendation:** `None` / `None` — string matching is deterministic, so a
numeric score conflates "rule fired" with "biological certainty"; leaving
both empty signals to consumers that a score isn't applicable to this
method. Confirm with curators.

### Q2. `data_novelty` value
The ML pipeline reads `data_novelty` from per-model metadata stored in ABC.
The string matching classifier has no such metadata, and we don't want
null values in this column. Available ATP terms:
- **`"ATP:0000335"`** ("data novelty") — the parent term in the ontology;
  same default as `agr_document_classifier_trainer.py:394`.
- **`"ATP:0000334"`** ("existing data") — leaf term.
- **`"ATP:0000321"`** ("new data") — leaf term.
- **`"ATP:0000228"`** ("new to database") — leaf term.
- **`"ATP:0000229"`** ("new to field") — leaf term.

A rule-based string matcher cannot semantically distinguish between
"new data" / "new to database" / "new to field" / "existing data" — that
classification requires reading the paper's claims, not just matching
antibody-related strings. So the leaf terms over-claim.

**Recommendation:** `"ATP:0000335"` ("data novelty", parent term),
matching the default in `agr_document_classifier_trainer.py:394`. It
declines to specify novelty subtype, which is the honest answer for a
deterministic regex pipeline. Confirm with curators.

### Q3. TET source row — method, description, ECO code
The source row for the TETs this pipeline emits has three fields curators
should review (`get_tet_source_id` in `utils/abc_utils.py:131`).

**a) `source_method`** (string, used in the API URL):
Proposed `"abc_antibody_string_matching_classifier"`. Alternative:
`"abc_string_matching_classifier"` (more reusable for future similar
pipelines). Sources are MOD-specific by URL, not topic-specific, so
either works.

**Recommendation:** keep the antibody-specific name. If we later add
other string-matching topic classifiers they can register their own
sources; mixing them under one source row would lose per-topic
attribution.

**b) `description`** (free text, shown to curators in the source listing):
Proposed `"Alliance string matching topic classification pipeline for
antibody"`. **This wording should come from the curators.** Daniela —
what would you like this row to say?

**c) `source_evidence_assertion`** (ECO code):
Proposed `"ECO:0008021"` ("string matching") to mirror what the existing
entity extractor uses (`abc_utils.py:133`). **TBC** — confirm with
curators that this is the correct code for a string-matching *topic
classifier* (vs. entity extractor) before the source is provisioned in
prod.

### Q4. Backfill scope — all WB in-corpus references, or a date-bounded subset?
The §5.2 backfill currently plans to add `ATP:0000366` ("antibody string
matching classification needed") to **every** WB in-corpus reference that
doesn't already have a tag in the antibody string-matching process. WB's in-corpus reference count is in the tens of
thousands; classifying all of them entails proportional TEI downloads,
text parsing, and TET POSTs.

Options:
- **All in-corpus references** — completest coverage; lets curators see
  string-matching results on the whole back catalog. Cost: large initial
  pipeline run, possibly creating TETs that conflict (or duplicate notes
  via the `" | "` merge logic) with manual curation already present on
  older papers.
- **Date-bounded** (e.g., references published in the last year, or since
  2019 — the cutoff used for the NN training set in SCRUM-4688) — focuses
  on papers most likely to need fresh curation. Cheaper and lower noise
  for curators. Downside: older papers added to the corpus *after* the
  rollout would still need attention via the auto-create chain (which
  fires regardless of pub date), so the date-bound is only an initial
  catch-up filter.
- **In-corpus *and* never previously antibody-curated** — the most
  surgical. Skips papers where Daniela's manual curation already lives,
  avoiding any TET note merging. Requires checking the WB-side antibody
  curation status, which we can pull via wbtools or a CSV from Daniela.

Cost concerns to factor in:
- Each backfill row inserts only ATP:0000366; the work happens later when
  the pipeline picks up jobs. So the backfill itself is cheap. The real
  cost is the downstream pipeline run.
- TEI files are cached per reference, so once per paper the GROBID PDF
  conversion is paid; re-runs on the same paper hit the cache.

**Recommendation pending input:** lean toward "last 5 years" (since 2020)
as a reasonable middle ground — recent enough that curators care, narrow
enough that the initial run completes in days rather than weeks. Confirm
with Daniela.

## 7. Files changed / created

### `agr_automated_information_extraction` (branch `SCRUM-5601`)

| Path | Change | Rough size |
|---|---|---|
| `agr_document_classifier/antibody_rules.py` | new | ~70 lines |
| `agr_document_classifier/agr_antibody_string_matching_classifier.py` | new | ~250 lines |
| `utils/abc_utils.py` | add `note` arg + eco_map entry | ~5 lines |
| `tests/agr_document_classifier/test_antibody_rules.py` | new | ~80 lines |
| `tests/agr_document_classifier/test_antibody_classifier_pipeline.py` | new | ~120 lines |
| `Makefile` | add `classify_antibody` target | ~3 lines (optional) |

### `agr_literature_service` (branch `SCRUM-5601`)

| Path | Change | Rough size |
|---|---|---|
| `lit_processing/oneoff_scripts/SCRUM-5601_add_antibody_str_match_transitions.py` | new — INSERTs four transitions, UPDATEs two `text conversion → file converted to text` action arrays | ~80 lines |
| `lit_processing/oneoff_scripts/SCRUM-5601_backfill_antibody_str_match_needed_tag.py` | new — backfill ATP:0000366 for existing WB in-corpus references | ~80 lines |

No changes to `mod_corpus_association_crud.py`, `workflow_tag_crud.py`, or
the `workflow/data/*.py` seed files — the existing text-conversion → file-
converted action chain handles auto-creation once the transitions oneoff
has registered the new rows. Tests for the workflow can be exercised end-
to-end on stage post-deploy; no new pytest fixtures required.

## 8. Rollout plan

1. Land `agr_literature_service` PR first.
2. On stage:
   a. Run `SCRUM-5601_add_antibody_str_match_transitions.py` to register
      the four new transitions and append the new `proceed_on_value`
      action to the two existing WB text-conversion rows.
   b. Verify by adding a fresh WB Experimental reference and confirming
      ATP:0000366 appears once the chain reaches "file converted to text".
   c. Run `SCRUM-5601_backfill_antibody_str_match_needed_tag.py`.
   d. Spot-check a sample of WB in-corpus references for ATP:0000366.
3. Land `agr_automated_information_extraction` PR; deploy.
4. Run the antibody pipeline on stage with `--stage --test_mode` against a
   few curated positive/negative WB papers; verify TETs and notes in the
   stage UI.
5. On production: run the transitions oneoff, then the backfill oneoff.
6. Schedule the pipeline (cron, same way the existing classifier runs).
7. Coordinate with Daniela on validation: compare a sample of new ABC TETs
   against the old WB DB output for the same papers, confirm parity.
