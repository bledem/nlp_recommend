from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pickle
import os
from nltk.corpus import stopwords

from nlp_recommend.utils.clean_data import tokenizer
from nlp_recommend.models.base import BaseModel
from nlp_recommend.settings import TOPK
from nlp_recommend.const import PARENT_DIR

MODEL_PATH = os.path.join(PARENT_DIR, 'weights/tfidf_model.pkl')
MAT_PATH = os.path.join(PARENT_DIR, 'weights/tfidf_mat.pkl')


class TfIdfModel(BaseModel):
    def __init__(self, data=None):
        super().__init__(name='TfIdf')
        self.load()
        if not hasattr(self, 'embed_mat') or not hasattr(self, 'model'):
            assert data is not None, 'No cache data found, add data argument'
            self.fit_transform(data)
        assert (self.model)

    def load(self):
        if os.path.exists(MAT_PATH):
            self.embed_mat = pickle.load(open(MAT_PATH, "rb"))
        if os.path.exists(MODEL_PATH):
            self.model = pickle.load(open(MODEL_PATH, "rb"))

    def fit_transform(self, data):

        stop_words = set(stopwords.words('english'))
        # , ngram_range=(1, 2)) #one gram to three gram
        vectorizer = TfidfVectorizer(
            stop_words=stop_words)
        # -> (num_sentences, num_vocabulary)
        tfidf_mat = vectorizer.fit_transform(data)
        self.embed_mat = tfidf_mat
        self.model = vectorizer

    def save_embeddings(self,
                        model_path=MODEL_PATH,
                        mat_path=MAT_PATH):
        """ """
        with open(mat_path, 'wb') as fw:
            pickle.dump(self.embed_mat, fw)

        with open(model_path, 'wb') as fw:
            pickle.dump(self.model, fw)

    def predict(self, in_sentence):
        tokens = [str(tok) for tok in tokenizer(in_sentence)]
        vec = self.model.transform(tokens)
        # Create list with similarity between two sentence
        mat = cosine_similarity(vec, self.embed_mat)
        best_index = self.extract_best_indices(mat, topk=TOPK)
        return best_index


if __name__ == '__main__':
    test_sentence = 'My life'
    tfidf_model = TfIdfModel()
    tfidf_model.predict(test_sentence)