from nlp_recommend.const import DATA_PATH, PARENT_DIR
from nlp_recommend.utils.clean_data import clean_beginning, clean_text, tokenizer, uppercase_eng
from nlp_recommend.settings import MAX_CHAR_PER_ROW
from nlp_recommend.models import SentimentCls
from nlp_recommend.settings import BATCH_SIZE
import numpy as np
import re
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import spacy

DATA_PATH = os.path.join(PARENT_DIR, 'training','dataset')


def apply_voc(vocab, char):
    if isinstance(char, list) or isinstance(char, np.ndarray):
        X = [vocab.doc2idx(sent) for sent in char]
    else:
        X = vocab.doc2idx[char]
    return X


class LoadData:
    """Load and Clean a DataFrame Dataset"""

    def __init__(self, dataset='philosophy',
                 cache=False, # if you have cleaned it already True
                 n_max=None,
                 random=False,
                 char_max=None,
                 remove_numbered_rows=True,
                 remove_person_row=True,
                 tokenize=True,
                 lemmatize=True,
                 max_prcs=6):
        self.dataset = dataset
        self.n_max = n_max  # maximum number of row
        self.random = random
        self.char_max = char_max if char_max else MAX_CHAR_PER_ROW
        self.remove_numbered_row = remove_numbered_rows
        self.remove_person_row = remove_person_row
        self.tokenize = tokenize
        self.lemmatize = lemmatize
        self.max_prcs = max_prcs
        self.data_path = os.path.join(DATA_PATH, dataset+'.csv')
        self.clean_data_path = os.path.join(
            DATA_PATH, dataset+'_clean.csv')
        self.tagged_data_path = os.path.join(
            DATA_PATH, dataset+'_tagged.csv')
        # Load already cleaned csv data file
        if cache and dataset:
            self.corpus_df = pd.read_csv(
                self.clean_data_path, lineterminator='\n', index_col=0)
            # Read tok_lem_sentence as lists and not strings
            # self.corpus_df['tok_lem_sentence'] = self.corpus_df['tok_lem_sentence'].apply(
            #     lambda x: eval(x))
        # Load and clean
        else:
            print('Lod data with ``load_and_clean`` method, save the results with ``save_dataset``.')
            self.load_and_clean()
            self.save_datasets()

    def load_and_clean(self):
        df = self._load_df()
        print(df.columns)
        df = self._clean_sentences(df)
        print(df.columns)
        # df = df.reset_index()
        self.corpus_df = df

    def _load_df(self):
        df = pd.read_csv(self.data_path, lineterminator='\n')
        df['valid'] = True #consider all sentences valid by default
        if self.random:
            df = df.sample(frac=1)
        if self.n_max:
            df = df.iloc[:self.n_max]
        df = self._clean_corpus(df)
        return df

    def _clean_corpus(self, df):
        df = df.dropna(subset=['sentence'])
        df = self._clean_titles(df)
        if self.remove_numbered_row:
            df = self._remove_numbers(df)
        if self.remove_person_row:
            df = self._remove_person(df, self.max_prcs)
        return df

    @staticmethod
    def _uppercase_eng(df):
        df.sentence['sentence'].apply
    @staticmethod
    def _clean_titles(df):
        pattern = re.compile(r'[\n\r\t]')
        df['title'] = df['title'].apply(lambda x: pattern.sub(' ', x))
        return df

    def _clean_sentences(self, df):
        print('Cleaning sentences...')
        df['sentence'] = df['sentence'].apply(uppercase_eng)
        df['small_clean_sentence'] = df['sentence'].apply(clean_beginning)
        df['clean_sentence'] = df['small_clean_sentence'].apply(clean_text)
        df['clean_sentence'] = df['clean_sentence'].apply(
            lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))
        df["tok_lem_sentence"] = df['clean_sentence'].apply(
            lambda x: tokenizer(x, lemmatize=self.lemmatize))
        # df = df.loc[(df['tok_lem_sentence'].str.len() < self.char_max)]
        df.loc[(df['tok_lem_sentence'].str.len() < self.char_max), 'valid'] = False
        return df

    def _remove_numbers(self, df):
        print('Removing numbers...')
        # df.reset_index(inplace=True)
        index_to_remove = [idx for idx, row in df.iterrows(
        ) if re.findall(r'[0-9]', row.sentence)]
        # df.drop(index=index_to_remove, inplace=True) # we don't remove but tag unvalid sentences
        df.loc[index_to_remove, 'valid'] = False
        return df

    @staticmethod
    def _remove_person(df, max_prcs):
        print('Removing proper noun...', len(df))
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError: #install if not already installed
            os.system('python -m spacy download en_core_web_sm')
            nlp = spacy.load("en_core_web_sm")

        list_result = []
        batchs = np.array_split(df, max(1, len(df)//BATCH_SIZE))

        with ProcessPoolExecutor(max_prcs) as e:
            fs = [e.submit(process_one_chunk, mini_df, nlp)
                  for mini_df in batchs]

            for f in tqdm(as_completed(fs), total=len(fs), desc='Processing'):
                if f._exception:
                    print(f._exception)
                assert f._exception is None
                list_result.append(f._result)
        df_result = pd.concat(list_result)
        # df_result = df_result.loc[~df_result.propn]
        # df_result = df_result.drop(columns=['propn'])
        df_result.loc[df_result.propn, 'valid'] = False #indicate as non valid the rows with proper noun
        df_result = df_result.drop(columns=['propn'])
        return df_result

    def save_datasets(self):
        """Save two datasets:
        1) Clean dataset (used for matching)
        2) Original dataset (used for warping)
        That share the same ``index`` and ``sent_index``
        """
        # Save cleaned reduced df
        df_clean = self.corpus_df.loc[self.corpus_df.valid]
        df_clean['index'] = df_clean.index
        df_clean.to_csv(self.clean_data_path, index=False)
        # Save original but tagged dataset
        self.corpus_df['index'] = self.corpus_df.index
        self.corpus_df.to_csv(self.tagged_data_path, index=False)



def detect_propn(x, nlp):
    tag = 'PROPN'
    pos_list = [w.pos_ for w in nlp(x)]
    return tag in pos_list


def process_one_chunk(df, nlp):
    df['propn'] = df.sentence.apply(lambda x: detect_propn(x, nlp))
    return df


if __name__ == '__main__':
    DATASET = 'philosophy'

    pd.options.display.max_rows = 1000
    pd.options.display.max_colwidth = 500

    # create the .csv for the first time
    # corpus = LoadData(dataset=DATASET,random=False, remove_numbered_rows=True, cache=False)
    # load the .csv
    corpus = LoadData(dataset=DATASET, cache=True) 
    philo_df = corpus.corpus_df

    philo_df = philo_df[['sentence', 'author', 'title', 'tok_lem_sentence']]
    philo_df['clean_sentence'] = philo_df['sentence'].apply(lambda x: clean_text(x, only_symbols=True))

    print('number of sentences:', len(philo_df))
    philo_df.head()