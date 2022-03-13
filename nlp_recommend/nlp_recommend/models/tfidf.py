from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pickle
import os
from nltk.corpus import stopwords

from nlp_recommend.utils.clean_data import tokenizer
from nlp_recommend.models.base import BaseModel
from nlp_recommend.settings import TOPK
from nlp_recommend.const import WEIGHT_DIR


class TfIdfModel(BaseModel):
    def __init__(self, data=None, dataset='philosophy', weight_dir=WEIGHT_DIR):
        super().__init__(name='TfIdf')
        self.dataset = dataset
        self.weight_dir = weight_dir
        self.model_path = os.path.join(
            weight_dir, 'weights', dataset, 'tfidf_model.pkl')
        self.mat_path = os.path.join(
            weight_dir, 'weights', dataset, 'tfidf_mat.pkl')
        self.load()
        if not hasattr(self, 'embed_mat') or not hasattr(self, 'model'):
            assert data is not None, f'No cache data found at {self.mat_path}, add data argument'
            self.fit_transform(data)
        assert (self.model)

    def load(self):
        if os.path.exists(self.mat_path):
            self.embed_mat = pickle.load(open(self.mat_path, "rb"))
        if os.path.exists(self.model_path):
            self.model = pickle.load(open(self.model_path, "rb"))

    def transform(self, data):
        return self.model.transform(data)

    def fit_transform(self, data):

        stop_words = set(stopwords.words('english'))
        # , ngram_range=(1, 2)) #one gram to three gram
        vectorizer = TfidfVectorizer(
            stop_words=stop_words)
        # -> (num_sentences, num_vocabulary)
        tfidf_mat = vectorizer.fit_transform(data)
        self.embed_mat = tfidf_mat
        self.model = vectorizer

    def save_embeddings(self):
        """ """
        with open(self.mat_path, 'wb') as fw:
            pickle.dump(self.embed_mat, fw)

        with open(self.model_path, 'wb') as fw:
            pickle.dump(self.model, fw)

    def predict_vec(self, in_sentence):
        tokens = [str(tok) for tok in tokenizer(in_sentence)]
        vec = self.model.transform(tokens)
        return vec

    def predict(self, in_sentence, topk=TOPK):
        vec = self.predict_vec(in_sentence)
        # Create list with similarity between two sentence
        mat = cosine_similarity(vec, self.embed_mat)
        best_index = self.extract_best_indices(mat, topk=topk)
        return best_index


if __name__ == '__main__':
    test_sentence = 'My life'
    tfidf_model = TfIdfModel()
    tfidf_model.predict(test_sentence)
