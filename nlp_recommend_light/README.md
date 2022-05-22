# Litterature recommendation system

## Getting started (with docker)
```
# download models and data folder
# copy this folder on your computer as <path_to_weights_folder>
# it contains a dataset and a weight folder. 
wget https://drive.google.com/drive/folders/1qsQ-QNN4gD_JoI2q4Q0Ew68LzKNdJyRH?usp=sharing
docker build -t nlp_recommend_auto .
docker run -v <path_to_weights_folder>:/training/ -p 5000:5000 nlp_recommend_auto
```

Debug:
```
docker run -it -v <path_to_weights_folder>:/training -p 5000:5000 -e FLASK_APP="nlp_recommend/app/webapp_light" nlp_recommend_auto bash
```


## Getting started (without docker) 
### 1. Set up your environment
```
pip install virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt 
```
### 2) Download the weights
```
wget https://drive.google.com/drive/folders/1qsQ-QNN4gD_JoI2q4Q0Ew68LzKNdJyRH?usp=sharing
```


### 3) Download the weights
Run the application
```
export FLASK_APP=nlp_recommend/app/app
export FLASK_ENV=development
flask run -h 0.0.0.0 -p 5000
```
Go to url: localhost:5000 in your browser. 



## Upgrade for better model

If you have 32Gb RAM Memory, you can try to load ``philosophy_container.pkl`` instead of ``philosophy_container_light.pkl`` in the module ``app.py``.

## Explore the code

### Auto 
```
docker run -v /Users/10972/Documents/NLP_PJ/models:/app/nlp_recommend/models -v /Users/10972/Documents/NLP_PJ/data:/app/data -p 5000:5000 nlp_recommend_auto
```
### Auto debug

Image size: 1.28Gb. CPU usage 5Gb.

```
make docker-build
make docker-run-auto 
```

Inference with spacy eng large model. 

