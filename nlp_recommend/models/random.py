from transformers.utils import logging
from nlp_recommend.models import BertModel, TfIdfModel, Word2VecModel, SpacyModel, CombinedModel
from nlp_recommend.const import DATASET_PATH
import os
import random
import logging

logger = logging.getLogger(__name__)


PROBA_DICT = {1: 'combined', 2: 'combined', 3: 'combined', 4: 'combined', 5: 'bert',
              6: 'spacy', 8: 'tfidf', 9: 'tfidf', 10: 'word2vec'}

MODEL_MAP = {'bert': BertModel, 'spacy': SpacyModel,
             'word2vec': Word2VecModel, 'tfidf': TfIdfModel, 'combined': CombinedModel}


class RandomModel():
    def __init__(self, dataset='philosophy', cls=None):
        self.dataset = dataset
        self.cls = cls
        self.data_path = os.path.join(DATASET_PATH, f'{dataset}_clean.csv')
        assert os.path.exists(self.data_path)

    def predict(self, sentence, history=[], proba_dict=PROBA_DICT):
        while True:
            if history:
                proba_dict = {k: v for k, v in proba_dict.items()
                              if v not in history}
            model_choice = proba_dict[random.randint(1, 10)]
            model = MODEL_MAP[model_choice]
            logger.info('Use model', model)
            index = self.predict_one_model(sentence, model=model)
            if len(index) > 0:
                index = index[0]  # take the best fit
                break
            else:
                history.append(model_choice)
        return index, model_choice

    def predict_one_model(self, sentence, model):
        model = model(dataset=self.dataset)
        index = model.predict(sentence)
        if self.cls:
            index, sentiment = self.cls.match_filter(sentence, index)
        return index
