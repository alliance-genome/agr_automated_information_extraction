# SCRUM-5764 — ZFIN molecular-probe abstract classification: evaluation results

**Date:** 2026-08-15 · **Ticket:** SCRUM-5764 · **Labelled set:** Ceri Van Slyke's
"Probe Training set-abstract classification" sheet (951 rows, exported 2026-08-11)

## Recommendation

**Ship the bag-of-words-only supervised classifier.** It matches or beats the
embedding-augmented model on every headline metric, classifies all 15
curator-annotated hard negatives correctly, and carries none of the operational
dependencies the embedding model needs.

Concretely, shipping BoW-only means: no OpenAI key in the classification pipeline,
no abstract embedding profile, no dependency on the SCRUM-6140 embedding pipeline,
and nothing extra to version in the ABC (the hashing vectorizer is stateless).

## Training-set coverage

| | |
|---|---|
| rows in sheet | 951 |
| duplicate xrefs dropped | 1 (`ZDB-PUB-260210-3`) |
| xrefs that failed to resolve | 0 |
| references with no abstract | **0** |
| **usable** | **950** (251 positive / 699 negative) |
| curator-annotated hard negatives | 15 |

**There is no recall ceiling from missing abstracts.** The plan anticipated that
references without an abstract in the ABC would be unclassifiable, capping
achievable recall and forcing manual triage for some fraction of the corpus. Every
labelled reference has an abstract (median 1,492 characters), so that concern does
not apply to this set.

Two data-quality notes, both of which silently corrupt a naive pipeline:

- Positive labels in the sheet read `"Positive "` — capital P **and a trailing
  space** — while the trainer compares `classification_value == "positive"` exactly.
  Uploaded unnormalized, every row lands in neither the positive nor the negative
  list and training sees an empty set.
- The `AGRKBID` column is empty for all 951 rows, so every reference must be
  resolved from its xref. The ABC's `by_cross_reference` endpoint requires the
  **MOD-prefixed** form (`ZFIN:ZDB-PUB-170126-4`); a bare `ZDB-PUB-170126-4`
  returns 404. See "Related defect" below.

## Method

Features are built with the production reader (`paragraph_pool_and_text`) over
ABC-format parquets, so what is measured here is what would ship. The BoW block is
a stateless hashed bag-of-words (2^18 features) over
`abstract_chunk_text(title, abstract)`; the embedding block is the L2-normalized
chunk-mean pool of `text-embedding-3-small` vectors (1536-d).

Two evaluations are reported:

1. **Single 80/20 stratified holdout** — reproduces the split
   `_select_and_fit_model` performs internally (`random_state=42`), with full
   `RandomizedSearchCV` model selection across 11 candidate classifiers. This is
   the production training path.
2. **Out-of-fold, 5-fold stratified, model class fixed to LGBMClassifier** — every
   one of the 950 references gets a held-out prediction.

The second evaluation exists because the single split placed only **2 of the 15**
hard negatives in the test set. A hard-negative specificity computed on n=2 is not
a result. Fixing the model class also isolates the effect of the *features* from
the effect of model selection, which otherwise confounds the comparison (the two
arms selected different classifiers).

## Results

### Single 80/20 holdout (190 references, 50 positive)

| arm | selected model | precision | recall | F1 | AP | CV F1 |
|---|---|--:|--:|--:|--:|--:|
| `bow_only` | LGBMClassifier | 1.000 | **0.980** | **0.990** | 1.000 | 0.983 |
| `embedding+bow` | LogisticRegression | 1.000 | 0.940 | 0.969 | 0.998 | 0.983 |

### Out-of-fold over all 950 (LGBMClassifier, like-for-like)

| arm | precision | recall | F1 | AP | recall@P95 | recall@P99 | hard negatives |
|---|--:|--:|--:|--:|--:|--:|--:|
| **`bow_only`** | **0.984** | **0.980** | **0.982** | **0.9986** | 0.996 @ 0.024 | 0.952 @ 0.869 | **0/15 wrong** |
| `embedding+bow` | 0.976 | 0.972 | 0.974 | 0.9974 | 0.984 @ 0.039 | 0.956 @ 0.953 | 0/15 wrong |

Confusion matrix for `bow_only` at the default 0.5 threshold:
**TP 246 · FP 4 · TN 695 · FN 5**.

### Does the embedding block help?

No. Out-of-fold with the model class held constant, adding 1,536 dense dimensions
costs 0.8 points of F1 and 0.8 points of recall. On the single split with full model
selection it cost 4 points of recall.

The differences are small — roughly 8 references out of 950 — and should be read as
**"embeddings provide no measurable benefit"** rather than "BoW is significantly
better." The decision between the two arms is therefore operational, not statistical,
and on operational grounds BoW-only wins decisively.

This reproduces the SCRUM-6052 finding on fulltext, where the embedding alone
underperformed BoW and embedding+BoW merely matched it. With 950 samples and 251
positives, a dense block of that width dilutes rather than adds.

## Hard negatives — the result that matters most

Ceri annotated 15 negatives with `string "probe" in abstract`: papers that mention a
probe but still contain curatable zebrafish biology. These are precisely the
references a keyword rule gets wrong.

**All 15 are classified correctly, out-of-fold.** Their predicted probabilities:

```
0.265, 0.003, 0.023, 0.000, 0.128, 0.016, 0.000, 0.005,
0.000, 0.000, 0.000, 0.095, 0.002, 0.017, 0.006
```

The highest is 0.265, far below any plausible operating threshold; most are
effectively zero.

This is the strongest evidence in the evaluation. The classes are lexically
near-disjoint — "probe" appears in 94% of positives versus 3% of negatives — so a
model that had merely learned the keyword would flag all 15 of these as positives.
It flags none. The model has learned the compositional *"only* discusses a probe"
distinction, which the design discussion predicted pooled embeddings would capture
poorly and an LLM might capture better.

| term in abstract | positives | negatives |
|---|--:|--:|
| "probe" | 94% | 3% |
| "fluoresc" | 95% | 8% |
| "detection" | 79% | 3% |
| "imaging" | 60% | 7% |
| "zebrafish" | 98% | 96% |

## Recommended operating threshold

The auto-tagging decision ("won't curate" / "won't manually index") needs a
precision-first threshold. Two candidates, both from the out-of-fold `bow_only` run:

| threshold | precision | recall | effect on 950 references |
|--:|--:|--:|---|
| **0.869** | 0.99 | 0.952 | 240 auto-flagged, **2 wrongly dropped**, 13 probe papers still reach triage |
| 0.024 | 0.95 | 0.996 | higher recall, ~5× the wrongly-dropped rate |

**Recommended: 0.869.** Wrongly dropping a curatable paper is the expensive error —
it is invisible, whereas a probe paper that survives to manual triage merely costs a
few seconds of a curator's time. At this threshold roughly 95% of this category
leaves the acquisition queue, at a cost of ~2 mistakes per 950 references.

This is a curator judgement, not an engineering one, and Ceri should make the call.

## Limitations

- **The negative set is curated, not sampled.** The 15 hard negatives were collected
  deliberately as adversarial examples, and only 2% of negatives mention "probe."
  In the live acquisition stream the hard-negative rate is likely higher, so
  real-world precision may be lower than 0.984. Worth re-measuring after the first
  production run.
- **951 references is a small set** (251 positives). Differences below ~1 point
  should not be treated as meaningful.
- The out-of-fold evaluation refits a single model class per fold rather than
  rerunning the full search; it is a robustness check on a specific slice, not the
  headline metric.

## Related defect (not part of this ticket)

`get_curie_from_xref` (`utils/abc_utils.py:534`) calls
`/reference/by_cross_reference/{xref}`, which matches on the **MOD-prefixed** curie.
`agr_dataset_manager/dataset_upload_from_csv.py:21` and
`agr_dataset_manager/dataset_downloader.py:83` pass the sheet's `XREF` column
through unprefixed, and treat the resulting 404 as "reference not found":

```python
agrkb_id = get_curie_from_xref(xref)
if not agrkb_id:
    logger.warning(f"Skipping invalid row: {row}")
    continue
```

On a sheet like this one — bare xrefs, empty `AGRKBID` — `create_dataset` succeeds
and then every entry is skipped, producing an **empty dataset with no error**.

**This does not block the upload for this ticket.** The uploader only reaches
`get_curie_from_xref` when `AGRKBID` is blank, so
`scripts/prepare_zfin_probe_training_set.py` now also writes `upload_ready.csv` with
the curie already resolved, the xref MOD-prefixed, and the label lowercased. Feeding
that file to `dataset_upload_from_csv.py` never touches the defective path, and skips
950 redundant API round-trips as a side effect.

The defect itself is still live for anyone who feeds a bare-xref sheet with an empty
`AGRKBID` column, and the two `agr_dataset_manager` scripts remain unfixed.

## Status of external dependencies

- **ATP term:** resolved. Ceri minted `ATP:0000370` on 2026-08-10, so dataset upload
  and model registration are unblocked.
- **Classification trigger:** resolved. Ceri confirmed "inside corpus" is the correct
  trigger — ZFIN papers get a ZDB-PUB ID at PubMed acquisition, and the ABC picks
  them up shortly after, before students are given the paper.
- **SCRUM-6140 (abstract embedding pipeline):** *no longer on the critical path.*
  Because the recommended model needs no embeddings, this classifier can ship
  without it. It remains relevant for SCRUM-5765 and the other low-value-paper
  categories.
- **`embedding_file` registration:** the ABC API exposes
  `GET /reference/embedding_file/{id}` only — verified against both prod and stage
  OpenAPI specs. Catalog rows are created ABC-internally
  (`embedding_file_crud.create_or_update`), so an uploaded parquet is inert without
  one. This constrains the design of SCRUM-6140 but does not block this ticket.

## Reproducing

```bash
# 1. Prepare the labelled set (resolves xrefs, fetches title+abstract)
set -a && . ./.env.rdsprod && set +a
python3 scripts/prepare_zfin_probe_training_set.py \
  -f probe_training_set.csv -o outdir

# 2. Supervised arms — BoW-only needs no embeddings and no OpenAI key
python3 scripts/evaluate_probe_classifiers.py \
  -i outdir/labelled_abstracts.json -p outdir/parquets -o outdir/results --skip-llm

# Optional: generate abstract embeddings for the embedding+bow arm
OPENAI_API_KEY=... python3 scripts/generate_abstract_embeddings.py \
  -i outdir/labelled_abstracts.json -o outdir/parquets
```

## Open items

- The `llm` arm (`gpt-5.4-nano`) has not been run. At P 0.984 / R 0.980 with all
  hard negatives correct, it cannot change the recommendation; it would only
  document that the comparison was made.
- Decide whether to fix the `by_cross_reference` prefix defect in the two
  `agr_dataset_manager` scripts, and whether that warrants its own ticket. Not
  blocking — `upload_ready.csv` routes around it.
- Upload `upload_ready.csv` to the ABC as a dataset under `ATP:0000370`, train the
  production model with `--embedding_profile` left at its default (BoW-only needs no
  abstract embeddings), and register it.
