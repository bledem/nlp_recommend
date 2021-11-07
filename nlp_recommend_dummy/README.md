# Litterature recommendation system

## Getting started (with docker)
```
# download models and data folder
wget https://drive.google.com/drive/folders/1deuu3GxXJzdFgpGyzSsuvIw7VvxYupRx?usp=sharing
wget https://drive.google.com/drive/folders/1lUuA8uWkYRFGuZwnapCh86F50sGBs0wu?usp=sharing
docker build -t nlp_recommend_auto .
docker run -v <path_to_models_folder>:/app/nlp_recommend/models -v <path_to_data_folder>:/app/data -p 5000:5000 nlp_recommend_auto
```
## Which data should I have locally? 
### For training
Download meta data
```
- data (for context in Warper)
    - gutenberg_philosophy
    - gutenberg_psychology
- nlp_recommend
    - dataset
    - models
    - labels
    - weights 
```
### For inference
Folder architecture:
- data (for context in Warper)
    - gutenberg_philosophy
    - gutenberg_psychology
- models

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
wget https://drive.google.com/drive/folders/1deuu3GxXJzdFgpGyzSsuvIw7VvxYupRx?usp=sharing
wget https://drive.google.com/drive/folders/1lUuA8uWkYRFGuZwnapCh86F50sGBs0wu?usp=sharing
```
Check you downloaded the models/ data folders. 
Data should be in parent folder and models inside nlp_recommend

- data
- nlp_recommend
    - nlp_recommend
    - models

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
```
docker run -it -v /Users/10972/Documents/NLP_PJ/models:/app/nlp_recommend/models -v /Users/10972/Documents/NLP_PJ/data:/app/data -p 5000:5000 nlp_recommend_auto /bin/bash
```
