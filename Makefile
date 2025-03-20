ifndef ENV_FILE
	ENV_FILE=.env
endif

include ${ENV_FILE}

train:
	docker-compose --env-file ${ENV_FILE} run agr_automated_information_extraction python agr_document_classifier.py --mode train --embedding_model_path /data/agr_document_classifier/BioWordVec.vec.bin --datatype_train $(DATATYPE) --mod_train $(MOD)

train_flysql:
    docker run --env-file ${ENV_FILE} -v /data/harvdev/alliance_gocd/agr_document_classifier:/data/harvdev/alliance_gocd/agr_document_classifier agr_document_classifier python agr_document_classifier.py --mode train --embedding_model_path /data/harvdev/alliance_gocd/agr_document_classifier/BioWordVec.vec.bin --datatype_train $(DATATYPE) --mod_train $(MOD)

classify:
	docker-compose --env-file ${ENV_FILE} run agr_automated_information_extraction python agr_document_classifier.py --mode classify --embedding_model_path /data/agr_document_classifier/BioWordVec.vec.bin

extract_entities:
	docker-compose --env-file ${ENV_FILE} run agr_automated_information_extraction python agr_entity_extractor.py