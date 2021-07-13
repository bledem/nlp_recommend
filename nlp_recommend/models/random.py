from nlp_recommend.models import BertModel, TfIdfModel, Word2VecModel, SpacyModel
from nlp_recommend.models import SentimentCls
from nlp_recommend.utils.clean_data import format_text

import pandas as pd
import random
import os

MODEL_MAP = {'bert': BertModel, 'spacy': SpacyModel,
             'word2vec': Word2VecModel, 'tfidf': TfIdfModel}

PROBA_DICT = {1: 'bert', 2: 'bert', 3: 'bert', 4: 'bert', 5: 'spacy',
              6: 'spacy', 7: 'spacy', 8: 'tfidf', 9: 'tfidf', 10: 'word2vec'}

DATA_PATH = '/home/bettyld/PJ/Documents/NLP_PJ/nlp_recommend/dataset/merged_clean.csv'
ORG_TXT_DIR = '/home/bettyld/PJ/Documents/NLP_PJ/data/gutenberg'


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
        result = self.dataset_csv[['sentence', 'author', 'title']].iloc[index]
        title, sentence = result.title, result.sentence
        wrapped_sentence = self.wrap(title, sentence)
        return result, wrapped_sentence, model_choice

    def wrap(self, title, sentence, offset=3):
        res = None
        txt_path = os.path.join(ORG_TXT_DIR, title.replace(' ', '_')+'.txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                list_lines = f.readlines()
                if len(list_lines) == 1:
                    reduced = format_text(list_lines)
                else:
                    reduced = list_lines
            sentence_idx = [index for index, s in enumerate(
                reduced) if sentence in s]
            if sentence_idx:
                sentence_idx = sentence_idx[0]
                res = reduced[sentence_idx-offset: sentence_idx+offset]
        return res

    def predict_one_model(self, sentence, model):
        index = model.predict(sentence)
        index, sentiment = self.cls.match_filter(sentence, index)
        return index
