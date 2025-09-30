# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Commands

### Document Classification
```bash
# Train a classifier
docker-compose run agr_automated_information_extraction python agr_document_classifier/agr_document_classifier_trainer.py \
  --mode train \
  --datatype_train <topic_ATP_ID> \
  --mod_train <mod_abbreviation> \
  --embedding_model_path /data/agr_document_classifier/BioWordVec.vec.bin

# Classify documents
docker-compose run agr_automated_information_extraction python agr_document_classifier/agr_document_classifier_classify.py \
  --embedding_model_path /data/agr_document_classifier/BioWordVec.vec.bin

# Using Make shortcuts
make train DATATYPE=<topic_ATP_ID> MOD=<mod_abbreviation>
make classify
```

### Entity Extraction
```bash
# Extract entities
docker-compose run agr_automated_information_extraction python agr_entity_extractor/agr_entity_extraction_pipeline.py
make extract_entities
```

### Testing & Linting
```bash
# Run tests
pytest tests/

# Run linting
python3 -m flake8 .       # Local flake8
make run-local-flake8

# Run type checking
make run-mypy              # Docker-based mypy
make run-local-mypy        # Local mypy
```

### Docker Build
```bash
# Full build (base + application)
make doc_classifier_full_build

# Application only build
make doc_classifier_build
docker-compose build
```

## High-Level Architecture

### Project Structure
The repository implements machine learning pipelines for biocuration in the Alliance of Genome Resources (AGR). It consists of three main subsystems:

1. **Document Classification** (`agr_document_classifier/`)
   - Classifies scientific references into biocuration topics (data types)
   - Uses word embeddings (BioWordVec) and various ML models (XGBoost, LightGBM, etc.)
   - Each classifier is specific to a MOD (Model Organism Database) and topic combination
   - Trained models are uploaded to and fetched from the ABC (Alliance Biocuration Collective) API

2. **Entity Extraction** (`agr_entity_extractor/`)
   - Extracts biological entities (genes, species, etc.) from references
   - Supports both TF-IDF based extraction and BERT-based deep learning models
   - Includes string matching and custom tokenization for biological text

3. **Dataset Management** (`agr_dataset_manager/`)
   - Downloads training data from ABC in TEI format
   - Manages embedding matrices and document processing
   - Handles data upload/download workflows

### Key Integration Points

- **ABC API Integration**: All document fetching, model storage, and result submission goes through the ABC REST API (`utils/abc_utils.py`)
- **TEI Document Processing**: Documents are handled in TEI (Text Encoding Initiative) XML format via `utils/tei_utils.py`
- **Authentication**: Uses Okta OAuth2 for API authentication (configured via environment variables)
- **Embedding Models**: Primarily uses BioWordVec for biological text embeddings, with support for other models

### MOD Abbreviations
The system works with these Model Organism Databases:
- WB (WormBase)
- MGI (Mouse Genome Informatics)
- SGD (Saccharomyces Genome Database)
- RGD (Rat Genome Database)
- ZFIN (Zebrafish Information Network)
- FB (FlyBase)

### Environment Configuration
Key environment variables (set in `.env`):
- `ABC_API_SERVER`: ABC API endpoint
- `OKTA_*`: OAuth2 authentication credentials
- `TRAINING_DIR`, `CLASSIFICATION_DIR`: Data directories
- `CLASSIFIERS_PATH`: Model storage location
- `GROBID_API_URL`: PDF-to-TEI conversion service

### Pipeline Workflow
1. Training: Downloads TEI documents → Extracts text → Generates embeddings → Trains classifier → Uploads model to ABC
2. Classification: Downloads documents → Fetches model from ABC → Classifies → Sends results back to ABC
3. Entity Extraction: Downloads documents → Applies extraction models → Tags entities → Submits to ABC

### Critical Dependencies
- `agr_literature_service`: Core AGR library for literature processing
- `fastapi_okta`: Authentication middleware
- ML frameworks: scikit-learn, XGBoost, LightGBM, transformers
- Text processing: NLTK, Gensim, fastText