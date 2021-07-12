
import numpy as np
import pickle
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

import sys


sys.path.insert(0, '/home/bettyld/PJ/Documents/NLP_PJ/philo')

CUR_DIR = os.path.dirname(__file__)

from load_data import LoadData
from clean_data import tokenizer


def load_model():
    feature_path = os.path.join(CUR_DIR, 'tfidf_mat.pkl')
    model_path = os.path.join(CUR_DIR,'tfidf.pkl')
    model = pickle.load(open(model_path, "rb"))
    mat = pickle.load(open(feature_path, "rb"))
    return model, mat

def load_database():
    corpus = LoadData(n_max=10000000000)
    corpus.load()
    philo_df = corpus.corpus
    philo_df = philo_df[['sentence', 'label', 'author', 'tok_lem_sentence']]
    philo_df = philo_df.loc[(philo_df.tok_lem_sentence.str.len()<10)]
    return philo_df

class TfidModel:
    def __init__(self):
        self.vectorizer, self.tfidf_mat = load_model()
        self.database = load_database()

    def process(self, text):
        tokens = [str(tok) for tok in tokenizer(text)]
        vec = self.vectorizer.transform(tokens)
        return vec

    def get_recommendations_tfidf(self, vec_sentence):
        """
        Return the database sentences in order of highest cosine similarity relatively to each 
        token of the target sentence. 
        """
        mat = cosine_similarity(vec_sentence, self.tfidf_mat)
        # best cos sim for each token independantly
        best_sim_each_token = np.argmax(mat, axis=1) 
        best_sim = np.max(mat, axis=1)
        index = np.argsort(best_sim_each_token)[::-1] #take the five highest norm 
        null_index = best_sim[index] > 0
        index = index[null_index]
        best_index = best_sim_each_token[index][:3]  
        return best_index

    def test_sentence(self, text):
        vect_text = self.process(text)
        best_index = self.get_recommendations_tfidf(vect_text)
        result = self.database[['sentence', 'author']].iloc[best_index].values
        return result

if __name__ == '__main__':
    model = TfidModel()
    res = model.test_sentence('This is a test')
    print(res)
