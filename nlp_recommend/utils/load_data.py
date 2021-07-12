from nlp_recommend.const import DATA_PATH, PARENT_DIR
from nlp_recommend.utils.clean_data import clean_text, tokenizer
from nlp_recommend.settings import MAX_CHAR_PER_ROW
from nlp_recommend.models import SentimentCls
from nlp_recommend.settings import BATCH_SIZE
import numpy as np
import re
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

CLEANDATA_PATH = os.path.join(PARENT_DIR, 'dataset/clean_sentences.csv')


def apply_voc(vocab, char):
    if isinstance(char, list) or isinstance(char, np.ndarray):
        X = [vocab.doc2idx(sent) for sent in char]
    else:
        X = vocab.doc2idx[char]
    return X


class LoadData:
    """An iterator to load the philo corpus"""

    def __init__(self, cache=False,
                 n_max=None,
                 random=False,
                 char_max=None,
                 remove_numbered_rows=True,
                 remove_person_row=True):
        self.n_max = n_max  # maximum number of row
        self.random = random
        self.char_max = char_max if char_max else MAX_CHAR_PER_ROW
        self.remove_numbered_row = remove_numbered_rows
        self.remove_person_row = remove_person_row
        # if cache:

    def load(self, tokenize=True, lemmatize=True):
        self.tokenize = tokenize
        self.lemmatize = lemmatize
        self.corpus = self.load_data()

        return self.corpus

    def load_data(self):
        df = self.load_philo()
        df = self.clean_sentences(df)
        df = df.reset_index()
        return df

    def load_philo(self, data_path=DATA_PATH):
        df = pd.read_csv(data_path)
        if self.random:
            df = df.sample(frac=1)
        if self.n_max:
            df = df.iloc[:self.n_max]
        df = self.clean_corpus(df)
        return df

    def clean_corpus(self, df):
        if self.remove_numbered_row:
            df = self.remove_numbers(df)
        if self.remove_person_row:
            df = self.remove_person(df)
        return df

    def clean_sentences(self, df):
        print('Cleaning sentences')
        df['clean_sentence'] = df['sentence'].apply(clean_text)
        df['clean_sentence'] = df['clean_sentence'].apply(
            lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))
        df["tok_lem_sentence"] = df['clean_sentence'].apply(
            lambda x: tokenizer(x, lemmatize=self.lemmatize))
        df = df.loc[(df['tok_lem_sentence'].str.len() < self.char_max)]
        return df

    def remove_numbers(self, df):
        print('Removing numbers...')
        df.reset_index(inplace=True)
        index_to_remove = [idx for idx, row in df.iterrows(
        ) if re.findall(r'[0-9]', row.sentence)]
        df.drop(index=index_to_remove, inplace=True)
        return df

    @staticmethod
    def remove_person(df):
        print('Removing proper noun...')
        import spacy
        nlp = spacy.load("en_core_web_sm")

        list_result = []
        batchs = np.array_split(df, len(df)//BATCH_SIZE)

        with ProcessPoolExecutor() as e:
            fs = [e.submit(process_one_chunk, mini_df, nlp)
                  for mini_df in batchs]

            for f in tqdm(as_completed(fs), total=len(fs), desc='Processing'):
                if f._exception:
                    print(f._exception)
                assert f._exception is None
                list_result.append(f._result)
        df_result = pd.concat(list_result)
        df_result = df_result.loc[df_result.propn]
        df_result = df_result.drop(columns=['propn'])
        return df_result


def detect_propn(x, nlp):
    tag = 'PROPN'
    pos_list = [w.pos_ for w in nlp(x)]
    return tag in pos_list


def process_one_chunk(df, nlp):
    df['propn'] = df.sentence.apply(lambda x: detect_propn(x, nlp))
    return df
