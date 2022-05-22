
IMAGE_NAME=nlp_recommend_light
TAG_NAME=15ed57f
CONTAINER_NAME=nlp_recommend

WEIGHT_DIR=/Users/10972/Documents/NLP_PJ
APP_PATH=nlp_recommend/app
FLASK_APP=$(APP_PATH)/app_light.py

.PHONY: docker-build docker-run clean-container clean

# Command for dockerization
docker-build:
	docker build -t $(IMAGE_NAME):$(TAG_NAME) .

docker-run-bash-debug:
	docker run --rm -it -v $(PWD):/app/nlp_recommend -v ${WEIGHT_DIR}/training:/app/training -p 5001:5001 --name=$(CONTAINER_NAME) $(IMAGE_NAME):$(TAG_NAME) /bin/bash

docker-run-auto:
	docker run --rm -it -v $(PWD):/app/nlp_recommend -v ${WEIGHT_DIR}/training:/app/training -p 5001:5001 --name=$(CONTAINER_NAME) $(IMAGE_NAME):$(TAG_NAME) \
	python nlp_recommend/app/webapp_light.py --data-path=/app/training --port 5001

clean-container:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

clean-image:
	-docker rmi $(IMAGE_NAME):$(TAG_NAME)

docker-push: 
	docker push bledem/$(IMAGE_NAME):$(TAG_NAME)

docker-create-models:
	#TODO
