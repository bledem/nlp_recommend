import spacy
import os
import numpy as np
import pickle

from nlp_recommend.model_deployment.models.base import BaseModel
from nlp_recommend.const import PARENT_DIR

MAT_PATH = os.path.join(PARENT_DIR, 'weights/spacy_mat.pkl')


class SpacyModel(BaseModel):
    def __init__(self, data=None):
        super().__init__(name='Spacy')
        self.load()
        if not hasattr(self, 'embed_mat'):
            print('generating...')
            self.fit_transform(data)
        assert (self.model)
        assert (self.embed_mat)

    def load(self):
        self.model = spacy.load("en_core_web_lg")
        if os.path.exists(MAT_PATH):
            self.embed_mat = pickle.load(open(MAT_PATH, "rb"))

    def fit_transform(self, data):
        # calling nlp on a string and spaCy tokenizes the text and creates a document object
        self.embed_mat = list(map(lambda x: self.model(' '.join(x)), data))

    def predict(self, in_sentence):
        doc_test = self.model(in_sentence)
        # print(doc_test)
        mat = np.array([doc_test.similarity(line) for line in self.embed_mat])
        # keep if vector has a norm
        mat_mask = np.array(
            [True if line.vector_norm else False for line in self.embed_mat])
        best_index = self.extract_best_indices(mat, topk=3, mask=mat_mask)
        return best_index

    def save_embeddings(self, mat_path=MAT_PATH):
        with open(mat_path, 'wb') as fw:
            pickle.dump(self.embed_mat, fw)


if __name__ == '__main__':

    test_sentence = 'My life'
    tfidf_model = SpacyModel()
    tfidf_model.predict(test_sentence)
