from transformers.utils import logging
from nlp_recommend.utils.clean_data import format_text
from nlp_recommend.utils.clean_data import clean_beginning
from nlp_recommend.const import DATASET_PATH, ORG_TXT_DIR

import pandas as pd
import random
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Warper:
    def __init__(self, dataset, offset=2, dataset_path=DATASET_PATH):
        """
        Find the sentence before and after one sentence is the corpus.
        Args:
            dataset (str): name of the dataset (should be in the parent folder's data dir as a suffix)
            offset (int): number of sentences (before and after) we add to the prediction to add context. 
        """
        self.dataset = dataset
        self.data_path = os.path.join(dataset_path, f'{dataset}_clean.csv')
        assert os.path.exists(self.data_path)
        self.corpus = pd.read_csv(self.data_path, lineterminator='\n')
        self.offset = offset

    def predict(self, model, sentence, return_index=False, topk=5):
        best_index = model.predict(sentence, topk=topk)
        result = self.corpus[['sentence', 'author', 'title']].iloc[best_index]
        title, sentence = result.title.values[0], result.sentence.values[0]
        wrapped_sentence = self._wrap(title, sentence, self.dataset, offset=2)
        if return_index:
            result = best_index
        return result, wrapped_sentence

    @staticmethod
    def _wrap(title, sentence, dataset, offset):
        res = {'before':None, 'sentence':None, 'after':None}
        trial = 5
        sentence_idx = []
        txt_path = os.path.join(
            ORG_TXT_DIR+f'_{dataset}', title.replace(' ', '_')+'.txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                list_lines = f.readlines()
                if len(list_lines) == 1:
                    reduced = format_text(list_lines)
                else:
                    reduced = list_lines
                    reduced = [s.strip() for s in reduced]
            reduced = clean_beginning(' '.join(reduced)).split('. ')
            # Looking for the sentence among the corpus
            # The sentence has been cleaned so it does not
            # match perfectly the original corpus
            delimiter = len(sentence) // trial
            while len(sentence_idx) != 1 and trial > 0:
                sentence_idx = [index for index, s in enumerate(
                    reduced) if sentence[:delimiter*trial] in s]
                trial -= 1
            if len(sentence_idx) == 1:
                sentence_idx = sentence_idx[0]
                res['before'] = reduced[sentence_idx-offset:sentence_idx]
                res['sentence'] = reduced[sentence_idx]
                res['after'] = reduced[sentence_idx+1:sentence_idx+offset]
        return res

if __name__ == '__main__':
    import sys
    from nlp_recommend import CombinedModel
    PARENT_DIR = '/Users/10972/Documents/NLP_PJ/nlp_recommend'
    dataset = 'psychology'
    test_sentence = 'I have received a beautiful flower today'
    warper = Warper(dataset=dataset)
    model = CombinedModel(dataset=dataset)
    warper.predict(model, test_sentence)