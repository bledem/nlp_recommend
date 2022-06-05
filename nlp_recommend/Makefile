
IMAGE_NAME=nlp_recommend
TAG_NAME=test_tag
CONTAINER_NAME=nlp_recommend_app

WEIGHT_DIR=/Users/10972/Documents/NLP_PJ/training
APP_PATH=nlp_recommend/app
FLASK_APP=$(APP_PATH)/webapp_light.py

.PHONY: docker-build docker-run clean-container clean

# Command for dockerization
docker-build:
	docker build -t $(IMAGE_NAME):$(TAG_NAME) .

docker-run-bash-debug:
	docker run --rm -it -v $(PWD):/app/nlp_recommend -v $(WEIGHT_DIR):/app/training -p 5000:5000 --name=$(CONTAINER_NAME) $(IMAGE_NAME):$(TAG_NAME) /bin/bash
# 	run inside the container:
# 	python nlp_recommend/app/webapp_light.py --data-path=/app/training --port 5001


docker-run-auto:
	docker run --rm -it -v $(PWD):/app/nlp_recommend -v ${WEIGHT_DIR}:/app/training -p 5000:5000 --name=$(CONTAINER_NAME) $(IMAGE_NAME):$(TAG_NAME) \
	python nlp_recommend/app/webapp_light.py --data-path=/app/training --port 5000

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
