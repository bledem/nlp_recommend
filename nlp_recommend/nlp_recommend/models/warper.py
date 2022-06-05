import logging
from nlp_recommend.const import WEIGHT_DIR

import pandas as pd
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Warper:
    def __init__(self, dataset, offset=3, dataset_path=WEIGHT_DIR):
        """
        Find the sentence before and after one sentence is the corpus.
        Args:
            dataset (str): name of the dataset (should be in the parent folder's data dir as a suffix)
            offset (int): number of sentences (before and after) we add to the prediction to add context. 
        """
        self.dataset = dataset
        self.data_path = os.path.join(dataset_path, 'dataset', f'{dataset}_tagged.csv') # valid+unvalid sentence
        assert os.path.exists(self.data_path)
        self.corpus = pd.read_csv(self.data_path, lineterminator='\n')
        self.offset = offset

    def predict(self, model, sentence, return_index=False, topk=5):
        best_valid_index = model.predict(sentence, topk=topk) # -> len(best_index) = topk (best index of the valid sentence)
        # simulate the valid corpus to retrieve correct index
        result = self.corpus[self.corpus.valid].iloc[best_valid_index] # -> df( topk rows x 9 columns )
        # top 1 title and sentence
        # title, sentence = result.title.values[0], result.sentence.values[0]
        wrapped_sentence = self.warp(result, offset=self.offset)
        if return_index:
            result = best_valid_index
        return result, wrapped_sentence

    def warp(self, result, offset):
        res = {'before':[], 'sentence':[], 'after':[]}
        for _, row in result.iterrows():
            before_idx = [row.org_idx-i for i in range(1, offset)]
            after_idx = [row.org_idx+i for i in range(1, offset)]
            res['before'].append(self.corpus.loc[self.corpus.org_idx.isin(before_idx), ['sentence', 'title', 'author']])
            res['after'].append(self.corpus.loc[self.corpus.org_idx.isin(after_idx), ['sentence', 'title', 'author']])
            res['sentence'].append(row[['sentence', 'title', 'author']])
        return res


if __name__ == '__main__':
    from nlp_recommend import SpacyModel
    dataset = 'psychology'
    test_sentence = 'I have received a beautiful flower today'
    warper = Warper(dataset=dataset)
    model = SpacyModel(dataset=dataset)
    idx, wrapped_sentence = warper.predict(model, test_sentence, return_index=True)
    print(wrapped_sentence)