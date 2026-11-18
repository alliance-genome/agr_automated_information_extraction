# Records which commit the build context is checked out at (SCRUM-6878), however the image
# is built; every job logs it when it starts (utils/build_info.py). Only the two small files
# reach the final image; .git does not.
FROM python:3.11-alpine3.20 AS build_info
RUN apk add --no-cache git
COPY .git /tmp/repo.git
RUN mkdir /build_info && \
    (git -c safe.directory='*' --git-dir=/tmp/repo.git rev-parse --short=8 HEAD || echo unknown) \
        > /build_info/commit && \
    date -u +%Y-%m-%dT%H:%M:%SZ > /build_info/build_date

FROM agr_document_classifier_base

ENV TZ=UTC
# Set here as well as in Dockerfile_Base: the base image is built separately by GoCD,
# so this file cannot assume the base it is built on already carries it.
ENV PYTHONUNBUFFERED=1
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /usr/src/app
ADD ./requirements.txt .
ADD utils/abc_utils.py .
ADD agr_document_classifier/agr_document_classification_pipeline.py .
ADD agr_document_classifier/agr_document_classifier_classify.py .
ADD agr_document_classifier/agr_document_classifier_trainer.py .
ADD agr_document_classifier/agr_priority_classifier_balanced.py .
# Antibody string-matching classifier (SCRUM-5601): copy as an importable package so
# `python3 -m agr_document_classifier.agr_antibody_string_matching_classifier` can resolve
# `from agr_document_classifier.antibody_rules import ...`.
ADD agr_document_classifier/__init__.py ./agr_document_classifier/__init__.py
ADD agr_document_classifier/antibody_rules.py ./agr_document_classifier/antibody_rules.py
ADD agr_document_classifier/agr_antibody_string_matching_classifier.py ./agr_document_classifier/agr_antibody_string_matching_classifier.py
ADD agr_dataset_manager/dataset_downloader.py .
ADD agr_dataset_manager/dataset_upload_from_csv.py .
ADD agr_entity_extractor/agr_entity_extraction_pipeline.py .
ADD agr_entity_extractor/agr_entity_extraction_pipeline_fast.py .
ADD agr_entity_extractor/agr_species_extraction_pipeline.py .
ADD agr_entity_extractor/fit_and_upload_tfidf_vectorizer.py .
ADD agr_entity_extractor/fit_and_upload_tfidf_vectorizer_fast.py .
ADD agr_entity_extractor/upload_string_matching_extractor.py .
ADD agr_entity_extractor/models.py ./agr_entity_extractor/models.py
ADD bert_entity_extraction ./bert_entity_extraction
ADD utils ./bert_entity_extraction/utils
ADD Makefile .
ADD utils/ ./utils
ADD agr_dataset_manager/ ./agr_dataset_manager
# POSSIBLE_CLASSIFIERS is imported as `agr_document_classifier.models` by the trainer and
# by scripts/evaluate_probe_classifiers.py, so it has to land inside the package here --
# the image otherwise has an importable agr_document_classifier/ with no models.py in it.
ADD agr_document_classifier/models.py ./agr_document_classifier/models.py
ADD export_data/export_fb_tets.py .
ADD export_data/export_fb_tets_using_score.py .
ADD bin/ ./bin/
ADD scripts/ ./scripts/
RUN chmod +x ./bin/run_export_and_commit.sh ./bin/check_textmining_freshness.sh ./bin/_send_report_shim.py
RUN apt-get -y install git-lfs wget
RUN apt-get update && apt-get install --no-install-recommends --yes build-essential git cron subversion
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
RUN git lfs install --force
RUN git clone https://huggingface.co/cgrivaz/FlyBaseGeneAbstractClassifier.git
RUN export PYTHONPATH=/usr/src/app/bert_entity_extraction:/usr/src/app/bert_entity_extraction/gene_finding:/usr/src/app/

# Which commit this image was built from (SCRUM-6878): read from .git by the build_info stage;
# GIT_SHA/BUILD_DATE build arguments take precedence when passed.
COPY --from=build_info /build_info /etc/abc_build_info
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
ENV GIT_SHA=${GIT_SHA} BUILD_DATE=${BUILD_DATE}
LABEL org.opencontainers.image.revision=${GIT_SHA} org.opencontainers.image.created=${BUILD_DATE}

CMD ["/bin/bash"]
