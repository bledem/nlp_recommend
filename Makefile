
IMAGE_NAME=nlp_recommend
TAG_NAME=20211001-b1ee09697
CONTAINER_NAME=nlp_recommend_20211003$(FLASK_APP)

WEIGHT_DIR=/Users/10972/Documents/NLP_PJ
APP_PATH=nlp_recommend/app

.PHONY: docker-build docker-run clean-container clean

# Command for dockerization
docker-build:
	cp $(APP_PATH)/$(FLASK_APP).py $(APP_PATH)/app_main.py
	docker build -t $(CONTAINER_NAME):$(TAG_NAME) .

docker-run-bash-debug:
	docker run --env-file $(ENV_FILE) -it -v $(DATA_FOLDER):/data -v --link=$(DB_LINK) $(CONTAINER_NAME):$(TAG_NAME) /bin/bash

docker-run-auto:
	docker run -v ${WEIGHT_DIR}/training:/app/training -p 5000:5000 $(CONTAINER_NAME):$(TAG_NAME)

docker-run-auto-light:
	docker run -v ${WEIGHT_DIR}/data:/app/training -p 5000:5000 $(CONTAINER_NAME):$(TAG_NAME)
	
docker-run-auto-debug:
	docker run -it -v ${WEIGHT_DIR}/nlp_recommend:/app/nlp_recommend -v ${WEIGHT_DIR}/training:/app/training -p 5000:5000 $(CONTAINER_NAME):$(TAG_NAME) /bin/bash

clean-container:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

clean-image:
	-docker rmi $(IMAGE_NAME):$(TAG_NAME)

docker-create-models:
	#TODO