from transformers import pipeline
import pickle
import os
import numpy as np
from tqdm import tqdm

# from nlp_recommend.const import PARENT_DIR

PARENT_DIR = '/home/bettyld/PJ/Documents/NLP_PJ/nlp_recommend'
MAT_PATH = os.path.join(PARENT_DIR, 'labels/sent_labels.pkl')
BATCH_SIZE = 100


class SentimentCls():
    def __init__(self, data=None):
        self.load()
        if not hasattr(self, 'labels'):
            assert data is not None, 'No cache data found, add data argument'
            self.fit(data)

    def load(self, mat_path=MAT_PATH):
        self.model = pipeline('sentiment-analysis')
        if os.path.exists(mat_path):
            self.labels = pickle.load(open(mat_path, 'rb'))

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
        in_label = self.predict(in_sentence)[0]['label']
        print(in_label)
        labels_idx = self.labels[idx_list]
        mask = np.argwhere(labels_idx == in_label).flatten()
        filtered = idx_list[mask]
        return filtered

    def predict(self, in_sentence):
        if in_sentence.__class__.__name__ == 'str':
            in_sentence = [in_sentence]
        elif not in_sentence.__class__.__name__ == 'list':
            in_sentence = list(in_sentence)
        label = self.model(in_sentence)
        return label

    def save_result(self, mat_path=MAT_PATH):
        with open(mat_path, 'wb') as fw:
            pickle.dump(self.labels, fw)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/bettyld/PJ/Documents/NLP_PJ/nlp_recommend')
    from nlp_recommend import LoadData

    corpus = LoadData(n_max=50, random=False, remove_numbered_rows=True)
    corpus.load()
    df = corpus.corpus
    cls = SentimentCls(data=df.sentence.values)
    index = [3, 4]
    filter_index = cls.match_filter('This is a trial', index)
    print(df.iloc[filter_index])
