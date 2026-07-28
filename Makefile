ifndef ENV_FILE
	ENV_FILE=.env
endif

include ${ENV_FILE}

# Exported so `make <target> APP_IMAGE=...` reaches docker-compose, which reads it
# to pick which image to run (prod by default, or the stage image). Empty is fine:
# compose falls back to its own default.
export APP_IMAGE

run-local-flake8:
	python3 -m flake8 .

run-mypy:
	docker-compose --env-file .env.test down -v
	docker-compose --env-file .env.test run -v ${PWD}:/workdir test_runner /bin/bash -c "mypy --config-file mypy.config agr_automated_information_extraction"
	docker-compose --env-file .env.test down -v

run-local-mypy:
	mypy --config-file mypy.config .

# GELF_COMPONENT feeds the gelf log driver's `tag` option in docker-compose.yaml, so
# each pipeline is identifiable in Graylog as agr.aie.<ENV_STATE>.<component>.
train:
	GELF_COMPONENT=train docker-compose --env-file ${ENV_FILE} run agr_automated_information_extraction python agr_document_classifier.py --mode train --embedding_model_path /data/agr_document_classifier/BioWordVec.vec.bin --datatype_train $(DATATYPE) --mod_train $(MOD)

classify:
	GELF_COMPONENT=classify docker-compose --env-file ${ENV_FILE} run agr_automated_information_extraction python agr_document_classifier.py --mode classify --embedding_model_path /data/agr_document_classifier/BioWordVec.vec.bin

extract_entities:
	GELF_COMPONENT=extract_entities docker-compose --env-file ${ENV_FILE} run agr_automated_information_extraction python agr_entity_extractor.py

classify_antibody:
	GELF_COMPONENT=classify_antibody docker-compose --env-file ${ENV_FILE} run agr_automated_information_extraction python -m agr_document_classifier.agr_antibody_string_matching_classifier

# Image naming follows the Alliance convention agr.literature.<component>.<role>.
# The legacy agr_document_classifier / agr_document_classifier_base tags are still
# applied to the same images so existing GoCD tasks keep working; drop them once
# every pipeline references the new names.
# IMAGE_APP_TEST is built INDEPENDENTLY, normally from a feature branch, and runs
# against the stage/test database via a different env file. It is deliberately a
# separate build rather than a second tag on the prod image: sharing a build would
# mean the next prod build silently re-pointed the -test tag at prod code.
IMAGE_BASE=agr.literature.automated_information_extraction.server-base
IMAGE_APP=agr.literature.automated_information_extraction.server
IMAGE_APP_TEST=agr.literature.automated_information_extraction.server-test

doc_classifier_full_build:
	docker build . -f Dockerfile_Base -t ${IMAGE_BASE} -t agr_document_classifier_base
	docker build . -t ${IMAGE_APP} -t agr_document_classifier

doc_classifier_build:
	docker build . -t ${IMAGE_APP} -t agr_document_classifier

# Stage/test image. GoCD passes --no-cache for reproducibility; omitted here so a
# local branch build does not reinstall torch every time. Run it with an env file
# holding the stage DB credentials, e.g.
#   make classify ENV_FILE=.env.stage APP_IMAGE=${IMAGE_APP_TEST}
doc_classifier_build_test:
	docker build . -t ${IMAGE_APP_TEST} -t agr_document_classifier_stage

flybert_build:
	docker build . -f Dockerfile_Base -t ${IMAGE_BASE} -t agr_document_classifier_base
	docker build . -t flybert
