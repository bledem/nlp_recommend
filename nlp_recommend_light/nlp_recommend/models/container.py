import sys
sys.path.insert(0, '/home/bettyld/PJ/Documents/NLP_PJ/nlp_recommend')

import pickle
import os

from nlp_recommend import SentimentCls, Warper, TfIdfModel


class ContainerVeryLight():
    def __init__(self, dataset, weight_dir, dataset_path):
        self.model = TfIdfModel(dataset=dataset, weight_dir=weight_dir)
        self.cls = SentimentCls(dataset=dataset, weight_dir=weight_dir)
        self.warper = Warper(dataset=dataset, dataset_path=dataset_path)

