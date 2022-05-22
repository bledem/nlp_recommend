import spacy
import os
import numpy as np
import pickle
from numpy import linalg as LA

from nlp_recommend.models.base import BaseModel
from nlp_recommend.const import PARENT_DIR, WEIGHT_DIR
from nlp_recommend.settings import TOPK

class SpacyModel(BaseModel):
    def __init__(self, data=None, dataset='philosophy', weight_dir=WEIGHT_DIR, empty=False):
        super().__init__(name='Spacy')
        self.dataset = dataset
        self.mat_path = os.path.join(
            weight_dir, 'weights', dataset, 'spacy_mat.pkl')
        self.load()
        if not hasattr(self, 'embed_mat') and not empty:
            print('generating embeddings...')
            self.fit_transform(data)
            assert (self.model)
            assert (self.embed_mat)
        elif empty:
            print('no weight loaded')

    def load(self):
        try:
            self.model = spacy.load("en_core_web_lg")
        except OSError:
            os.system('python3 -m spacy download en_core_web_lg')
            self.model = spacy.load("en_core_web_lg")
        if os.path.exists(self.mat_path):
            self.embed_mat = pickle.load(open(self.mat_path, "rb"))

    def fit_transform(self, data):
        # calling nlp on a string and spaCy tokenizes the text and creates a document object
        self.embed_mat = list(map(lambda x: self.model(x).vector, data))

    def transform(self, sentence):
        return self.model(sentence).vector

    def predict_vec(self, in_sentence, return_vec=True):
        res = self.model(in_sentence)
        if return_vec:
            res = res.vector
        return res

    @staticmethod
    def similarity(a, b):
        result = np.dot(a, b) / (LA.norm(a) * LA.norm(b))
        return result

    def predict(self, in_sentence, topk=TOPK):
        doc_test = self.predict_vec(in_sentence, return_vec=True)
        # mat = np.array([doc_test.similarity(line) for line in self.embed_mat])
        mat = np.array([self.similarity(doc_test, line) for line in self.embed_mat])

        # keep if vector has a norm
        mat_mask = np.array(
            # [True if line.vector_norm else False for line in self.embed_mat])
            [False for line in self.embed_mat])

        best_index = self.extract_best_indices(mat, topk=topk, mask=mat_mask)
        return best_index

    def save_embeddings(self):
        os.makedirs(os.path.dirname(self.mat_path), exist_ok=True)
        with open(self.mat_path, 'wb') as fw:
            pickle.dump(self.embed_mat, fw)


if __name__ == '__main__':
    import pandas as pd

    test_sentence = 'My life'
    genres = ['philosophy', 'adventure', 'psychology']
    for genre_ in genres:
        data_path = f'/Users/10972/Documents/NLP_PJ/training/dataset/{genre_}_clean.csv'
        clean_df = pd.read_csv(data_path)
        model = SpacyModel(dataset=genre_, data=clean_df.sentence.values)
        model.save_embeddings()
    # model.predict(test_sentence)
