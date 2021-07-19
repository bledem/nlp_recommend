from nlp_recommend.models.base import BaseModel
from transformers.utils import logging
from nlp_recommend.models import BertModel, TfIdfModel, Word2VecModel, SpacyModel
from nlp_recommend.models import SentimentCls, CombinedModel
from nlp_recommend.utils.clean_data import format_text
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import random
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

DATA_PATH = '/home/bettyld/PJ/Documents/NLP_PJ/nlp_recommend/dataset/merged_clean.csv'
ORG_TXT_DIR = '/home/bettyld/PJ/Documents/NLP_PJ/data/gutenberg'
MODEL_MAP = {'bert': BertModel, 'spacy': SpacyModel,
             'word2vec': Word2VecModel, 'tfidf': TfIdfModel}


class WarpedModel(CombinedModel):
    def __init__(self, dataset='merged', models=['spacy', 'bert']):
        super().__init__(dataset='merged', models=['spacy', 'bert'])
        # self.cls = SentimentCls(dataset=dataset)
        self.corpus = pd.read_csv(DATA_PATH, lineterminator='\n')

    def predict(self, sentence):
        best_index = super().predict(sentence)
        result = self.corpus[['sentence', 'author', 'title']].iloc[best_index]
        title, sentence = result.title.values[0], result.sentence.values[0]
        wrapped_sentence = self._wrap(title, sentence)
        return result, wrapped_sentence

    @staticmethod
    def _wrap(title, sentence, offset=3):
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
