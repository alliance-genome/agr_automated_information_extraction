FROM python:3.11-slim

ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /usr/src/app
ADD ./requirements.txt .
ADD utils/abc_utils.py .
ADD agr_document_classifier/agr_document_classification_pipeline.py .
ADD agr_document_classifier/agr_document_classifier_classify.py .
ADD agr_document_classifier/agr_document_classifier_trainer.py .
ADD agr_document_classifier/agr_priority_classifier_balanced.py .
ADD agr_dataset_manager/dataset_downloader.py .
ADD agr_dataset_manager/dataset_upload_from_csv.py .
ADD agr_entity_extractor/agr_entity_extraction_pipeline.py .
ADD agr_entity_extractor/agr_strain_extraction_pipeline.py .
ADD agr_entity_extractor/fit_and_upload_tfidf_vectorizer.py .
ADD agr_entity_extractor/fit_and_upload_tfidf_vectorizer_for_strain.py .
ADD agr_entity_extractor/upload_string_matching_extractor.py .
ADD agr_entity_extractor/models.py ./agr_entity_extractor/models.py
ADD Makefile .
ADD utils/ ./utils
ADD agr_dataset_manager/ ./agr_dataset_manager
ADD agr_document_classifier/models.py .
ADD export_data/export_fb_tets.py .
RUN apt-get update && apt-get install --no-install-recommends --yes build-essential git cron
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
CMD ["/bin/bash"]
