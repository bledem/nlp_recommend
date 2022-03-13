
IMAGE_NAME=nlp_recommend
TAG_NAME=8cee4b8
CONTAINER_NAME=nlp_recommend_20211003$(FLASK_APP)

WEIGHT_DIR=/Users/10972/Documents/NLP_PJ
APP_PATH=nlp_recommend/app
FLASK_APP=$(APP_PATH)/app_light.py

.PHONY: docker-build docker-run clean-container clean

# Command for dockerization
# Command for dockerization
docker-build:
	docker build -t $(CONTAINER_NAME):$(TAG_NAME) .

docker-run-bash-debug:
	docker run --env-file $(ENV_FILE) -it -v $(DATA_FOLDER):/data -v --link=$(DB_LINK) $(CONTAINER_NAME):$(TAG_NAME) /bin/bash

docker-run-auto:
	docker run -v ${WEIGHT_DIR}/training:/app/training -p 5000:5000 --env FLASK_APP=$(FLASK_APP) $(CONTAINER_NAME):$(TAG_NAME)
# docker run -p 5000:5000 --env FLASK_APP=$(FLASK_APP) $(CONTAINER_NAME):$(TAG_NAME)

docker-run-auto-light:
	docker run -p 5001:5001 $(CONTAINER_NAME):$(TAG_NAME)

docker-run-auto-debug:
	echo $(FLASK_APP)
	docker run -it -v ${WEIGHT_DIR}/training:/app/training -p 5000:5000 --env FLASK_APP=$(FLASK_APP) $(CONTAINER_NAME):$(TAG_NAME) /bin/bash

docker-run-auto-mount:
	echo $(FLASK_APP)
	docker run -it -v ${WEIGHT_DIR}/nlp_recommend:/app/nlp_recommend -v ${WEIGHT_DIR}/training:/app/training -p 5000:5000 --env FLASK_APP=$(FLASK_APP) $(CONTAINER_NAME):$(TAG_NAME) /bin/bash

clean-container:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

clean-image:
	-docker rmi $(IMAGE_NAME):$(TAG_NAME)

docker-push: 
	docker push bledem/$(IMAGE_NAME):$(TAG_NAME)

docker-create-models:
	#TODO
