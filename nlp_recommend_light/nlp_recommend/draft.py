import os

CUR_DIR = os.path.dirname(__file__) # app folder
PARENT_DIR = os.path.dirname(CUR_DIR) # nlp_recommend
MODEL_DIR = os.path.dirname(PARENT_DIR)

import sys
sys.path.insert(0, PARENT_DIR)

from nlp_recommend.models.container import Container

model_psycho = Container(dataset='psychology')
print('loaded!', model_psycho.warper.predict(model_psycho.model, 'test'))
