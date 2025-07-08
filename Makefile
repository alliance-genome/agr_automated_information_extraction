ifndef ENV_FILE
	ENV_FILE=.env
endif

include ${ENV_FILE}

run-local-flake8:
	python3 -m flake8 .

run-mypy:
	docker-compose --env-file .env.test down -v
	docker-compose --env-file .env.test run -v ${PWD}:/workdir test_runner /bin/bash -c "mypy --config-file mypy.config agr_automated_information_extraction"
	docker-compose --env-file .env.test down -v

run-local-mypy:
	mypy --config-file mypy.config .

train:
	docker-compose --env-file ${ENV_FILE} run agr_automated_information_extraction python agr_document_classifier.py --mode train --embedding_model_path /data/agr_document_classifier/BioWordVec.vec.bin --datatype_train $(DATATYPE) --mod_train $(MOD)

classify:
	docker-compose --env-file ${ENV_FILE} run agr_automated_information_extraction python agr_document_classifier.py --mode classify --embedding_model_path /data/agr_document_classifier/BioWordVec.vec.bin

extract_entities:
	docker-compose --env-file ${ENV_FILE} run agr_automated_information_extraction python agr_entity_extractor.py

extraction_build_full:
    docker build -f Dockerfile_Base -t entity_extraction_base
    docker build . -t agr_automated_information_extraction

extraction_build:
    docker build . -t agr_automated_information_extraction
