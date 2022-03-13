
from nlp_recommend.utils import LoadData
from nlp_recommend.models import SentimentCls, BertModel
from nlp_recommend.const import WEIGHT_DIR
from nlp_recommend.settings import BERT_BATCH_SIZE

def main():
    # create the .csv for the first time
    corpus = LoadData(dataset=DATASET,random=False, remove_numbered_rows=True, cache=False,  weight_dir=WEIGHT_DIR)

    # add a sentiment classification column
    df = corpus.corpus_df
    SentimentCls(dataset=DATASET, data=df, weight_dir=WEIGHT_DIR)
    
    # generate one model pkl
    _ = BertModel(dataset=DATASET, topk=3,
                       bert_model='sentence-transformers/paraphrase-mpnet-base-v2',
                       small_memory=True,
                       device='cpu', batch_size=BERT_BATCH_SIZE, weight_dir=WEIGHT_DIR)
    


if __name__ == '__main__':
    DATASET = 'philosophy'

    
