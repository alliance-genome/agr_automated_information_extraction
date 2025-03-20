# AGR Automated Information Extraction Models and Pipelines

This is the main Alliance repository for automated information extraction pipelines and models.

- Classify Alliance references into biocuration topics using machine learning models.
- Extract entities from references

The repository contains code to train models and upload them to the ABC. When used to classify new documents, the classifier related to the specified MOD abbreviation and topic (data type) is fetched from the ABC. 
Documents for training and classification are fetched from the ABC repository in TEI format. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Development](#development)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/agr_automated_information_extraction.git
    cd agr_automated_information_extraction
    ```

2. Create and configure the `.env` file:
    ```sh
    cp .env.example .env
    # Edit the .env file to include your specific configuration
    ```
   
3. Build the Docker image:
    ```sh
    docker-compose build
    ```


## Usage

### Training a Classifier

To train a classifier, run the following command:
```sh
docker-compose run agr_document_classifier python agr_document_classifier.py --mode train --datatype_train <topic_ATP_ID> --mod_train <mod_abbreviation> --embedding_model_path <path_to_embedding_model>
```

### Optional Arguments for training

- --weighted_average_word_embedding: Use weighted average for word embeddings.
- --standardize_embeddings: Standardize the embeddings.
- --normalize_embeddings: Normalize the embeddings.
- --sections_to_use: Specify sections to use for training.
- --skip_training_set_download: Skip downloading the training set.
- --skip_training: Skip the training process and upload a pre-existing model.


### Classifying Documents

To classify documents, run the following command:
```sh
docker-compose run agr_document_classifier python agr_document_classifier.py --mode classify --embedding_model_path <path_to_embedding_model>
```

### Configuration

The project uses environment variables for configuration. These variables are defined in the .env file. Key variables include:  

- TRAINING_DIR: Directory for training data.
- CLASSIFICATION_DIR: Directory for documents to classify.
- CLASSIFIERS_PATH: Path to save classifiers.
- GROBID_API_URL: URL for the GROBID API.
- ABC_API_SERVER: URL for the ABC API server.
- OKTA_*: Configuration for Okta authentication.
- CLASSIFICATION_BATCH_SIZE: Batch size for document classification.
