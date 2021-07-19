from nlp_recommend.models.base import BaseModel
from transformers.utils import logging
from nlp_recommend.models import BertModel, TfIdfModel, Word2VecModel, SpacyModel
from nlp_recommend.models import SentimentCls
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


class CombinedModel(BaseModel):
    def __init__(self, dataset='merged', models=['spacy', 'bert']):
        # self.cls = SentimentCls(dataset=dataset)
        # self.corpus = pd.read_csv(DATA_PATH, lineterminator='\n')
        self.models = {m: MODEL_MAP[m]() for m in models}
        self.concat_embed()

    def concat_embed(self):
        embed_mat = []
        if 'spacy' in self.models:
            embed_mat.append(self._get_spacy_embed())
        if 'bert' in self.models:
            embed_mat.append(self.models['bert'].embed_mat)
        if 'tfidf' in self.models:
            embed_mat.append(self.models['tfidf'].embed_mat)
        embed_mat = np.concatenate(embed_mat, axis=1)
        self.embed_mat = embed_mat

    def _get_spacy_embed(self):
        embed_mat = self.models['spacy'].embed_mat
        array_mat = np.zeros(
            (len(embed_mat), len(embed_mat[0].vector)))
        for idx, i in enumerate(embed_mat):
            array_mat[idx] = i.vector
        return array_mat

    def predict(self, sentence):
        input_vec = []
        for model in self.models:
            input_embed = self.models[model].transform(sentence)
            if model == 'bert':
                input_embed = input_embed[0]
            input_embed = np.array(input_embed)
            input_vec.extend(input_embed)
        input_vec = np.expand_dims(input_vec, axis=0)
        mat = cosine_similarity(input_vec, self.embed_mat)
        best_index = self.extract_best_indices(mat, topk=1)

        # result = self.corpus[['sentence', 'author', 'title']].iloc[index]
        # title, sentence = result.title, result.sentence
        # wrapped_sentence = self._wrap(title, sentence)
        # return result, wrapped_sentence, model_choice
        return best_index
