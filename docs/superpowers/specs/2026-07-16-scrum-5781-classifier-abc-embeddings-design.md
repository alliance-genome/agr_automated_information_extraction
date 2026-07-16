# SCRUM-5781 — Consume ABC precomputed embeddings in the document classifier

**Ticket:** SCRUM-5781 "Retrain FB models on new full text" (epic SCRUM-5990 "Update
pipelines after embedding db instantiated").

**Goal:** Let the document-classifier trainer and classifier use the OpenAI
embeddings the ABC now produces per reference (SCRUM-6141/6142), instead of
computing BioWordVec mean-pooled vectors on the fly — while keeping every model
already in production working exactly as before.

## Background (what the ABC already provides)

- After PDF→MD conversion, the ABC generates one **embedding parquet per merged
  Markdown** and registers it as a `referencefile` with `file_class="embedding"`,
  `file_extension="parquet"` (producer: `embedding_generation.py`).
- Exactly one profile exists today:
  - `profile_name = classifier_fulltext_paragraph_chunk_refs_excluded_md_cleaned`
  - `version = 1`, `model = text-embedding-3-small`, `dim = 1536`
- The parquet has one row per **paragraph chunk** plus one **document-level** row
  (`is_document_level == True`). Columns are the shared contract in
  `agr_abc_document_parsers.embeddings.parquet_io` (`embedding` is `list<float32>`).
- Discovery: `GET /reference/referencefile/show_all/{curie}` lists embedding rows,
  each enriched with `profile_name`, `version`, `model_name`, and a `source`
  pointer to the `converted_merged_*` Markdown it was computed from.
- Download: `GET /reference/referencefile/download_file/{referencefile_id}`
  (already wrapped by `get_file_from_abc_reffile_obj`).
- FB training-set references are already embedded (SCRUM-6144 Done).

## Decisions (confirmed with PO)

1. **Dense feature = L2-normalized chunk-mean pool** of the main-PDF paragraph
   embeddings (each paragraph vector L2-normalized, averaged, mean L2-normalized;
   `is_document_level` row excluded) — the SCRUM-6052 recipe.
2. **Concatenate the hashed BoW block** (`--use_bow_features`) over the parquet's
   own references-excluded paragraph text. Rationale (SCRUM-6052): the embedding
   alone underperformed BoW; embedding+BoW matched the BoW baseline. BoW text is
   taken from the parquet content (not a separate MD download) so classify needs
   only the parquet; this is a deliberate, transparent deviation from the
   prototype's MD-fulltext BoW with negligible expected impact.
3. **Date threshold** `--filter_date_before 2005-01-01`; **no outlier removal**
   (PO found it gave no advantage).
4. **Main PDF only:** select the embedding whose `source.file_class ==
   "converted_merged_main"`. Supplement embeddings are ignored.
5. **Retrocompat via a per-model marker** (not a global date cutoff): the trainer
   stamps a machine-readable marker into the model's ABC `ml_model.description`,
   including `bow=true/false`. The classifier reads it per model — marker present ⇒
   ABC-embedding path (rebuilding the identical embedding[+BoW] vector); absent ⇒
   the existing BioWordVec on-the-fly path, byte-for-byte unchanged. No ABC schema
   change: `description` already exists on `/ml_model/upload` and `get_model_data`.
6. **Scope of retraining:** the five FB document-classification topics with a
   training dataset — disease (`ATP:0000152`), new transgene (`ATP:0000013`), new
   allele (`ATP:0000006`), physical interaction (`ATP:0000069`), and "no genetic
   data" (`ATP:0000207`). Read datasets + embeddings from prod (the only place FB
   embeddings currently exist); upload to stage with `production=false`.

## Components

### New: `utils/abc_embeddings.py`
Single source of truth for the profile constants + marker format + parquet read.

- Constants: `ABC_EMBEDDING_PROFILE`, `ABC_EMBEDDING_VERSION`,
  `ABC_EMBEDDING_MODEL`, `ABC_EMBEDDING_DIM`, `ABC_EMBEDDING_POOLING="paragraph_mean"`.
- `format_embedding_marker(...) -> str` / `parse_embedding_marker(description) -> dict|None`:
  a sentinel-prefixed `key=value` marker, tolerant parsing. Detection is by the
  sentinel token; absence ⇒ `None`.
- `paragraph_mean_from_parquet(parquet_bytes) -> np.ndarray|None`: read the parquet
  from bytes with `pyarrow`, average the `embedding` of rows where
  `is_document_level` is false. `None` when there are no paragraph rows.

### `utils/abc_utils.py`
- `get_reference_embedding(reference_curie, mod_abbreviation, profile_name, version) -> np.ndarray|None`:
  `show_all` → pick the `embedding` row with the matching `profile_name`/`version`
  whose `source.file_class == "converted_merged_main"` → download parquet →
  `paragraph_mean_from_parquet`. `None` when the reference has no such embedding.
- `upload_ml_model(...)`: add an optional `description` parameter, sent as a form
  field in the upload payload.

### Trainer `agr_document_classifier_trainer.py`
- New flag `--use_abc_embeddings`. When set:
  - `--embedding_model_path` is not required (no BioWordVec load).
  - Build `X` from the ABC paragraph-mean vector of each positive/negative curie
    (references with no embedding are dropped and logged — same policy as a
    missing MD today). No MD download, no text pipeline, no BoW/max/LSH blocks.
  - Stamp the marker into `description` and pass it to `upload_ml_model`.
- The existing BioWordVec path is untouched when the flag is off.

### Classifier `agr_document_classifier_classify.py`
- Per model, after `get_model_data`, parse `description` for the marker.
  - Marker present ⇒ for each reference in the batch fetch the ABC paragraph-mean
    vector and predict on it; a reference with no embedding is treated as an
    invalid embedding (job failed) exactly like a missing MD today.
  - Marker absent ⇒ current BioWordVec path, unchanged.
- The BioWordVec embedding model is still loaded once for the mixed run; ABC-marked
  models ignore it.

## Data flow (ABC path)

```
curie ─ show_all ─▶ embedding row (profile match, source=converted_merged_main)
      ─ download_file ─▶ parquet bytes
      ─ paragraph_mean_from_parquet ─▶ 1536-d vector  ─▶ X row
```

## Dependencies
- Add `pyarrow` to `requirements.txt` (parquet reader). No `openai`/`tiktoken`
  needed on the consumer side (reading only). Install `pyarrow` in the cervino
  conda env before training.

## Cross-environment upload (operational)
The FB embeddings currently exist only on **prod** (verified 2026-07-16: prod
10/10 sample references embedded, stage 0/10), while the FB datasets exist on both
with the same `dataset_id`. To honor "train + upload to stage" without embeddings
on stage, `upload_ml_model` honors an optional `ABC_UPLOAD_API_SERVER` env var that
redirects **only** the model upload. A run therefore reads datasets + embeddings
from prod (`ABC_API_SERVER=…prod…`) and uploads the model to stage
(`ABC_UPLOAD_API_SERVER=…stage…`), all with `production=false`.

## Testing
- Unit tests with a synthetic parquet (pyarrow) and mocked `show_all`/download:
  - `paragraph_mean_from_parquet` averages only paragraph rows; `None` on
    doc-level-only.
  - `get_reference_embedding` selects the `converted_merged_main` profile row and
    ignores supplements / wrong profile / missing embedding.
  - marker round-trip (`format`/`parse`), and absence ⇒ `None` (retrocompat gate).

## Out of scope
- WB retraining (SCRUM-5780), abstract embeddings, paragraph-level (non-pooled)
  features, and any change to the ABC producer.
