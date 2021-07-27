import spacy
import os
import numpy as np
import pickle

from nlp_recommend.models.base import BaseModel
from nlp_recommend.const import PARENT_DIR
from nlp_recommend.settings import TOPK


class SpacyModel(BaseModel):
    def __init__(self, data=None, dataset='philosophy'):
        super().__init__(name='Spacy')
        self.dataset = dataset
        self.mat_path = os.path.join(
            PARENT_DIR, 'weights', dataset, 'spacy_mat.pkl')
        self.load()
        if not hasattr(self, 'embed_mat'):
            print('generating embeddings...')
            self.fit_transform(data)
        assert (self.model)
        assert (self.embed_mat)

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
        self.embed_mat = list(map(lambda x: self.model(' '.join(x)), data))

    def transform(self, sentence):
        return self.model(sentence).vector

    def predict(self, in_sentence, topk=TOPK):
        doc_test = self.model(in_sentence)
        mat = np.array([doc_test.similarity(line) for line in self.embed_mat])
        # keep if vector has a norm
        mat_mask = np.array(
            [True if line.vector_norm else False for line in self.embed_mat])
        best_index = self.extract_best_indices(mat, topk=topk, mask=mat_mask)
        return best_index

    def save_embeddings(self):
        with open(self.mat_path, 'wb') as fw:
            pickle.dump(self.embed_mat, fw)


if __name__ == '__main__':

    test_sentence = 'My life'
    tfidf_model = SpacyModel()
    tfidf_model.predict(test_sentence)
