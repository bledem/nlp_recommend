import sys
sys.path.insert(0, '/home/bettyld/PJ/Documents/NLP_PJ/nlp_recommend')

import pickle
import os

from nlp_recommend import BertModel, CombinedModel, SentimentCls, Warper
from nlp_recommend.const import PARENT_DIR

class Container():
    def __init__(self, dataset):
        self.model = CombinedModel(dataset=dataset)
        self.cls = SentimentCls(dataset=dataset)
        self.warper = Warper(dataset=dataset)


class ContainerLight():
    def __init__(self, dataset):
        self.model = BertModel(dataset=dataset)
        self.cls = SentimentCls(dataset=dataset)
        self.warper = Warper(dataset=dataset)



