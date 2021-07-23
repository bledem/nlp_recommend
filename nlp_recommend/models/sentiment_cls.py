from transformers import pipeline
import pickle
import os
import numpy as np
from tqdm import tqdm

from nlp_recommend.const import PARENT_DIR
from nlp_recommend.settings import BATCH_SIZE


class SentimentCls():
    def __init__(self, dataset='philosophy', data=None):
        self.label_path = os.path.join(
            PARENT_DIR, f'labels/{dataset}/labels.pkl')
        self.dataset = dataset
        self.load()
        if not hasattr(self, 'labels'):
            assert data is not None, 'No cache data found, add data argument'
            self.fit(data)

    def load(self):
        self.model = pipeline('sentiment-analysis')
        if os.path.exists(self.label_path):
            self.labels = pickle.load(open(self.label_path, 'rb'))

    def fit(self, data):
        self.labels = []
        for sub_data in tqdm(np.array_split(data, BATCH_SIZE)):
            if len(sub_data) == 0:
                break
            sub_data = sub_data.tolist()
            self.labels.extend([e['label'] for e in self.model(sub_data)])
        self.labels = np.array(self.labels)

    def match_filter(self, in_sentence, idx_list):
        """ Take the input sentence and return the dataset idx
        that matches with the input sentence sentiment."""
        # Retrieve the label of the input sentence
        sentiment = self.predict(in_sentence)[0]['label']
        labels_idx = self.labels[idx_list]
        mask = np.argwhere(labels_idx == sentiment).flatten()
        filtered = np.array(idx_list)[mask]
        return filtered, sentiment

    def predict(self, in_sentence):
        if in_sentence.__class__.__name__ == 'str':
            in_sentence = [in_sentence]
        elif not in_sentence.__class__.__name__ == 'list':
            in_sentence = list(in_sentence)
        label = self.model(in_sentence)
        return label

    def save(self):
        os.makedirs(os.path.dirname(self.label_path), exist_ok=True)
        with open(self.label_path, 'wb') as fw:
            pickle.dump(self.labels, fw)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/bettyld/PJ/Documents/NLP_PJ/nlp_recommend')
    from nlp_recommend import LoadData

    corpus = LoadData(dataset='philosophy', n_max=50,
                      random=False, remove_numbered_rows=True)
    corpus.load()
    df = corpus.corpus
    cls = SentimentCls(dataset='philosophy', data=df.sentence.values)
    index = [3, 4]
    filter_index = cls.match_filter('This is a trial', index)
    print(df.iloc[filter_index])
