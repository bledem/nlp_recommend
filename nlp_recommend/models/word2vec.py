from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
import numpy as np
import os
import pickle

from nlp_recommend.models.base import BaseModel
from nlp_recommend.settings import TOPK
from nlp_recommend.const import PARENT_DIR, WEIGHT_DIR


class Word2VecModel(BaseModel):
    def __init__(self, data=None, dataset='philosophy',
     fast_complete=False, weight_dir=WEIGHT_DIR):
        super().__init__(name='Word2Vec')
        self.dataset = dataset
        self.fast_complete = fast_complete
        self.model_path = os.path.join(
            weight_dir, 'weights', dataset, f'w2v.model')
        self.model_fast_path = os.path.join(
            weight_dir, 'weights', dataset, f'w2v_fast.model')
        self.mat_path = os.path.join(
            weight_dir, 'weights', dataset, f'w2v_mat.pkl')
        self.load()
        if not hasattr(self, 'dataset') or not hasattr(self, 'model'):
            assert data is not None, 'No cache data found, add data argument'
            self.fit_transform(data)

    def load(self):
        if os.path.exists(self.model_path):
            self.model = Word2Vec.load(self.model_path)
        if os.path.exists(self.mat_path):
            self.dataset = pickle.load(open(self.mat_path, 'rb'))
        if self.fast_complete and os.path.exists(self.model_fast_path):
            self.model_fast.save(self.model_fast_path)

    def fit_transform(self, data):
        self.model = Word2Vec(min_count=1,
                              workers=8,
                              vector_size=256)

        self.model.build_vocab(list(data))
        self.model.train(data,
                         total_examples=self.model.corpus_count,
                         epochs=10)

        self.dataset = data
        if self.fast_complete:
            self.model_fast = FastText(vector_size=256,
                                       window=3,
                                       min_count=1,
                                       sentences=data,
                                       epochs=10)

    def predict(self, sentence, topk=TOPK):
        sentence = sentence.split()
        in_vocab_list, best_index = [], [0]*topk
        # finding synonyms
        for w in sentence:
            if self.is_word_in_model(w):
                in_vocab_list.append(w)
            elif self.fast_complete:
                list_synonyms = self.find_synonym(w)
                for synonym in list_synonyms:
                    if self.is_word_in_model(synonym):
                        in_vocab_list.append(synonym)
                        break
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

    def save_embeddings(self):
        self.model.save(self.model_path)
        with open(self.mat_path, 'wb') as fw:
            pickle.dump(self.dataset, fw)
        if hasattr(self, 'model_fast'):
            self.model_fast.save(self.model_fast_path)
