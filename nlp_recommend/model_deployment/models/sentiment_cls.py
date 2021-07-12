from transformers import pipeline
import pickle
import os
import numpy as np

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
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
        for sub_data in np.array_split(data, BATCH_SIZE):
            sub_data = sub_data.tolist()
            self.labels.extend([e['label'] for e in self.model(sub_data)])
        self.labels = np.array(self.labels)

    def match(self, in_sentence, idx_list):
        in_label = self.predict(in_sentence)
        labels_idx = self.labels[idx_list]
        mask = np.argwhere(labels_idx == in_label)
        return mask

    def predict(self, in_sentence):
        if not in_sentence.__class__.__name__ == 'list':
            in_sentence = in_sentence.tolist()
        label = self.model(in_sentence)
        return label

    def save_result(self, mat_path=MAT_PATH):
        with open(mat_path, 'wb') as fw:
            pickle.dump(self.labels, fw)


if __name__ == '__main__':
    pass
