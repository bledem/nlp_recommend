# nlp_recommend

## 1. Set up your environment

pip install virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt 


## 2. Download pre-trained weights

### Download models

https://drive.google.com/drive/folders/1gtR-5KXIfm_LFCHO9z8ReoMTd09B-QAR?usp=sharing

Copy the directory ``models`` in the folder.

Architecture:

    Readme.md
    requirements.txt
    models
    nlp_recommend

## 3. Run the app

(you need to export the variable in CLI every time you launch the app)

export FLASK_APP=app/app/py
export FLASK_ENV=development
flask run -h localhost -p 5000

Go to url: localhost:5000 in your browser. 

## Upgrade for better model

If you have 32Gb RAM Memory, you can try to load ``philosophy_container.pkl`` instead of ``philosophy_container_light.pkl`` in the module ``app.py``.