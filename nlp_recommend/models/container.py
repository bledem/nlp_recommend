import sys
sys.path.insert(0, '/home/bettyld/PJ/Documents/NLP_PJ/nlp_recommend')

import pickle
import os

from nlp_recommend import BertModel, CombinedModel, SentimentCls, Warper
from nlp_recommend.const import WEIGHT_DIR, DATASET_PATH

class Container():
    def __init__(self, dataset, weight_dir=WEIGHT_DIR, dataset_path=DATASET_PATH):
        self.weight_dir = weight_dir
        self.model = CombinedModel(dataset=dataset, weight_dir=weight_dir)
        self.cls = SentimentCls(dataset=dataset, weight_dir=weight_dir)
        self.warper = Warper(dataset=dataset, dataset_path=dataset_path)


class ContainerLight():
    def __init__(self, dataset, weight_dir=WEIGHT_DIR, dataset_path=DATASET_PATH):
        self.model = BertModel(dataset=dataset, weight_dir=weight_dir)
        self.cls = SentimentCls(dataset=dataset, weight_dir=weight_dir)
        self.warper = Warper(dataset=dataset, dataset_path=dataset_path)

