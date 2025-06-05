# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the AGR (Alliance of Genome Resources) automated information extraction system - a bioinformatics ML pipeline for automated literature curation. It performs document classification and entity extraction on scientific literature to support biocuration workflows across multiple Model Organism Databases (MODs): WB, MGI, SGD, RGD, ZFIN, FB.

## Architecture

**Core Packages:**
- `agr_document_classifier/` - Document classification using embeddings and ML models (LogisticRegression, RandomForest, XGBoost)
- `agr_entity_extractor/` - Entity extraction using TF-IDF vectorization and string matching
- `agr_dataset_manager/` - Data management and downloading from ABC (Alliance Bibliography Corpus)
- `utils/` - Shared utilities for ABC API, TEI parsing, embeddings

**External Dependencies:**
- ABC API (Alliance Bibliography Corpus) for document storage/retrieval
- GROBID service for PDF to TEI conversion
- OKTA for authentication
- BioWordVec for biological word embeddings

## Development Commands

**Setup:**
```bash
docker-compose build
cp .env.example .env  # Configure ABC_API_SERVER, OKTA credentials
```

**Model Training:**
```bash
make train DATATYPE=<topic_ATP_ID> MOD=<mod_abbreviation>
```

**Classification:**
```bash
make classify
```

**Entity Extraction:**
```bash
make extract_entities
```

**Code Quality:**
```bash
make run-local-flake8    # Linting
make run-mypy           # Type checking
pytest tests/           # Run tests
```

## Key Configuration

**Required Environment Variables:**
- `ABC_API_SERVER` - Alliance Bibliography Corpus API endpoint
- `GROBID_API_URL` - PDF conversion service
- `OKTA_*` - Authentication credentials
- `CLASSIFICATION_BATCH_SIZE` - Processing batch size

**Docker Volume Mounts:**
- `${TRAINING_DIR}:/data/agr_document_classifier/training` - Training datasets
- `${CLASSIFICATION_DIR}:/data/agr_document_classifier/to_classify` - Documents to classify
- `${CLASSIFIERS_PATH}:/data/agr_document_classifier` - Trained models storage
- `${ENTITY_EXTRACTION_DIR}:/data/agr_entity_extraction/to_extract` - Entity extraction input

## Bioinformatics Context

- Understand MOD abbreviations: WB (C. elegans), MGI (mouse), SGD (yeast), RGD (rat), ZFIN (zebrafish), FB (fly)
- Gene ontology (GO) terminology and biological entity patterns
- TEI (Text Encoding Initiative) format for scientific document processing
- Curation workflows involve classification by biological topics and extraction of genes/proteins

## Testing

- Uses pytest framework with tests in `tests/` directory
- Focus on custom tokenizer and string matching functionality
- Test data uses biological entities (e.g., C. elegans gene names: lin-12, ced-3)