    """Script to generate the <dataset>_clean_sent.csv dataframe with original and valid data with sentiments
    """
from nlp_recommend.models import SentimentCls
from nlp_recommend.utils import LoadData
from nlp_recommend.const import WEIGHT_DIR

if __name__ == '__main__':
    dataset = 'adventure'
    # corpus = LoadData(dataset=dataset, # n_max=50,
    #                   random=False, remove_numbered_rows=True, cache=True)
    corpus = LoadData(dataset=dataset,random=False, remove_numbered_rows=True, cache=True)
    
    # df = corpus.corpus_df
    # cls = SentimentCls(dataset=dataset,data=df, weight_dir=WEIGHT_DIR)
    cls = SentimentCls(dataset=dataset, weight_dir=WEIGHT_DIR)
    index = [3, 4]
    filtered_df = cls.match_filter('This is a trial', index)
    print(filtered_df)
