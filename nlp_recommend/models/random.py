from nlp_recommend.models import BertModel, TfIdfModel, Word2VecModel, SpacyModel
from nlp_recommend.models import SentimentCls
import pandas as pd
import random

MODEL_MAP = {'bert': BertModel, 'spacy': SpacyModel,
             'word2vec': Word2VecModel, 'tfidf': TfIdfModel}

PROBA_DICT = {1: 'bert', 2: 'bert', 3: 'bert', 4: 'bert', 5: 'spacy',
              6: 'spacy', 7: 'spacy', 8: 'tfidf', 9: 'tfidf', 10: 'word2vec'}

DATA_PATH = '/home/bettyld/PJ/Documents/NLP_PJ/nlp_recommend/dataset/merged_clean.csv'


class RandomModel():
    def __init__(self, dataset='merged'):
        self.cls = SentimentCls(dataset=dataset)
        self.dataset_csv = pd.read_csv(DATA_PATH, lineterminator='\n')

    def predict(self, sentence, history=[], proba_dict=PROBA_DICT):
        while True:
            if history:
                proba_dict = {k: v for k, v in proba_dict.items()
                              if v not in history}
            model_choice = proba_dict[random.randint(1, 10)]
            model = MODEL_MAP[model_choice]()
            index = self.predict_one_model(sentence, model=model)
            if len(index) > 0:
                index = index[0]  # take the best fit
                break
            else:
                history.append(model_choice)
        result = self.dataset_csv[['sentence', 'author']].iloc[index]
        return result, model_choice

    def wrap:
        pass

    def predict_one_model(self, sentence, model):
        index = model.predict(sentence)
        index, sentiment = self.cls.match_filter(sentence, index)
        return index
