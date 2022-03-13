import os

CUR_DIR = os.path.dirname(__file__) # app folder
PARENT_DIR = os.path.dirname(CUR_DIR) # nlp_recommend
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(PARENT_DIR)), 'training')
import dill

import sys
sys.path.insert(0, PARENT_DIR)

from nlp_recommend.models.container import Container

model = dill.load(open(os.path.join(MODEL_DIR, 'models/philosophy_container_verylight.pkl'), 'rb'))
model.warper.predict(
        model.model, 'I am a spider', return_index=True, topk=5)

model_psycho = Container(dataset='psychology')
print('loaded!', model_psycho.warper.predict(model_psycho.model, 'test'))
