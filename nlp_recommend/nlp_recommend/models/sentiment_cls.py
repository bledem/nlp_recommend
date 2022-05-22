
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd 
from nlp_recommend.const import WEIGHT_DIR
from nlp_recommend.settings import BERT_BATCH_SIZE


class SentimentCls():
    def __init__(self, model=None, dataset='philosophy', data=None, weight_dir=WEIGHT_DIR, sentence_col='sentence', n_workers=5):
        self.label_path = os.path.join(
            weight_dir, f'dataset/{dataset}_clean_sent.csv')
        self.dataset = dataset
        self.sentence_col = sentence_col
        self.n_workers = n_workers
        self.model = model
        if os.path.exists(self.label_path):
            self.labels =  pd.read_csv(self.label_path)
        elif data is not None: # create self.labels as original data + sentiment
            if 'sentiment' not in data.columns:
                self.fit(data)
                self.save()

    def load(self):
        if os.path.exists(self.label_path):
            self.labels = pd.read_csv(self.label_path)

    def predict_sentiment(self, data):
        data['sentiment'] = data[self.sentence_col].apply(lambda x: self.model(x)[0]['label'])
        return data
    
    def fit(self, data):
        list_result = []
        batchs = np.array_split(data, max(1, len(data)//BERT_BATCH_SIZE))

        with ProcessPoolExecutor(self.n_workers) as e:
            fs = [e.submit(self.predict_sentiment, sub_data)
                  for sub_data in batchs]

            for f in tqdm(as_completed(fs), total=len(fs), desc='Processing'):
                if f._exception:
                    print(f._exception)
                assert f._exception is None
                list_result.append(f._result)
        self.labels = pd.concat(list_result)

    def match_filter(self, in_sentence, idx_list) -> pd.DataFrame:
        """ Take the input sentence and return the dataset idx
        that matches with the input sentence sentiment."""
        if self.model is not None:
        # Retrieve the label of the input sentence
            sentiment = self.predict(in_sentence)[0]['label'] # -> 'POSITIVE' or 'NEGATIVE'
            labels_idx = self.labels.iloc[idx_list]
            filtered = labels_idx.loc[labels_idx.sentiment==sentiment]
        else:
            filtered = self.labels.iloc[idx_list]
        return filtered

    def predict(self, in_sentence):
        if in_sentence.__class__.__name__ == 'str':
            in_sentence = [in_sentence]
        elif not in_sentence.__class__.__name__ == 'list':
            in_sentence = list(in_sentence)
        label = self.model(in_sentence)
        return label

    def save(self):
        os.makedirs(os.path.dirname(self.label_path), exist_ok=True)
        self.labels.to_csv(self.label_path, index=False)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/Users/10972/Documents/NLP_PJ/nlp_recommend/nlp_recommend')
    from nlp_recommend import LoadData
    dataset='adventure'
    corpus = LoadData(dataset=dataset, # n_max=50,
                      random=False, remove_numbered_rows=True, cache=True)
    df = corpus.corpus_df
    # cls = SentimentCls(dataset=dataset, data=df, weight_dir=WEIGHT_DIR)
    cls = SentimentCls(dataset='philosophy', weight_dir=WEIGHT_DIR)
    index = [3, 4]
    filter_index = cls.match_filter('This is a trial', index)
    # print(df.iloc[filter_index])
