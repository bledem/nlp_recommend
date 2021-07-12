from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
import numpy as np
import os
import pickle

from nlp_recommend.models.base import BaseModel
from nlp_recommend.settings import TOPK
from nlp_recommend.const import PARENT_DIR

MODEL_PATH = os.path.join(PARENT_DIR, 'weights/w2v.model')
MODEL_FAST_PATH = os.path.join(PARENT_DIR, 'weights/w2v_fast.model')
MAT_PATH = os.path.join(PARENT_DIR, 'weights/w2v_mat.pkl')


class Word2VecModel(BaseModel):
    def __init__(self, data=None):
        super().__init__(name='Word2Vec')
        self.load()
        if not hasattr(self, 'dataset') or not hasattr(self, 'model'):
            assert data is not None, 'No cache data found, add data argument'
            self.fit_transform(data)

    def load(self, mat_path=MAT_PATH, model_path=MODEL_PATH):
        if os.path.exists(mat_path):
            self.model = Word2Vec.load(MODEL_PATH)
        if os.path.exists(model_path):
            self.dataset = pickle.load(open(mat_path, 'rb'))
        # self.model_fast.save(MODEL_FAST_PATH)

    def fit_transform(self, data):
        self.model = Word2Vec(min_count=1,
                              workers=8,
                              vector_size=256)

        self.model.build_vocab(list(data))
        self.model.train(data,
                         total_examples=self.model.corpus_count,
                         epochs=10)

        self.dataset = data
        # self.model_fast = FastText(vector_size=256,
        #                            window=3,
        #                            min_count=1,
        #                            sentences=data,
        #                            epochs=10)

    def predict(self, sentence, topk=TOPK):
        sentence = sentence.split()
        in_vocab_list, best_index = [], [0]*topk
        # finding synonyms
        for w in sentence:
            if self.is_word_in_model(w):
                in_vocab_list.append(w)
    #         else:
    #             list_synonyms = find_synonym(w)
    #             for synonym in list_synonyms:
    #                 if is_word_in_model(synonym, wv):
    #                     in_vocab_list.append(synonym)
    #                     added += 1
    #                     break
        # print(in_vocab_list)
        if len(in_vocab_list) > 0:
            sim_mat = np.zeros(len(self.dataset))  # TO DO
            for i, data_sentence in enumerate(self.dataset):
                if data_sentence:
                    sim_sentence = self.model.wv.n_similarity(
                        in_vocab_list, data_sentence)
                else:
                    sim_sentence = 0
                sim_mat[i] = np.array(sim_sentence)
            # take the five highest norm
            best_index = np.argsort(sim_mat)[::-1][:topk]
            # print(sim_mat[best_index])
        return best_index

    def is_word_in_model(self, word):
        assert type(self.model.wv).__name__ == 'KeyedVectors'
        is_in_vocab = word in self.model.wv.key_to_index.keys()
        return is_in_vocab

    def find_synonym(self, word):
        list_similars = self.model_fast.wv.most_similar(
            positive=[word], topn=5)  # ([(w, cos_sim), ...])
        return [w[0] for w in list_similars]

    def save_embeddings(self, model_path=MODEL_PATH, mat_path=MAT_PATH,
                        model_fast_path=MODEL_FAST_PATH):
        self.model.save(model_path)
        with open(mat_path, 'wb') as fw:
            pickle.dump(self.dataset, fw)

        # self.model_fast.save(model_fast_path)
