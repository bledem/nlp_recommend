"""
Script to generate all pickles file for weights, sentiment labels 
from one dataset.
"""
from dataclasses import dataclass
import sys
import os

PARENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PARENT_DIR)

import dill 
from nlp_recommend.utils.load_data import LoadData
from nlp_recommend.models.container import Container, ContainerLight
import logging
logging.basicConfig(level=logging.DEBUG)

DATASET = 'philosophy'

logging.info(f'Loading data for {DATASET}')
corpus = LoadData(dataset=DATASET, cache=True)
philo_df = corpus.corpus_df
logging.info('Corpus loaded')

def generate_cls(dataset=DATASET, df=philo_df):
    from nlp_recommend.models import SentimentCls
    cls = SentimentCls(dataset=dataset, data=df.sentence.values)
    logging.info(f'Saving CLS labels')
    cls.save()
    
def generate_embed(model_name, dataset=DATASET, df=philo_df):
    logging.info(f'Loading {model_name}')
    if model_name == 'spacy':
        from nlp_recommend.models.spacy import SpacyModel
        model = SpacyModel(dataset=DATASET, data=df['tok_lem_sentence'])
    elif model_name == 'bert':
        from nlp_recommend.models.bert import BertModel
        model = BertModel(dataset=DATASET, data=df.sentence.values, device='cpu')
    logging.info(f'Saving {model_name} embeddings')
    model.save_embeddings()


def main(save_at, dataset=DATASET, light=False):
    if light:
        container = ContainerLight(dataset=dataset)
        saving_at = os.path.join(save_at, f'{dataset}_container_light.pkl')
    else:
        container = Container(dataset=dataset)
        saving_at = os.path.join(save_at, f'{dataset}_container.pkl')
    with open(saving_at, 'wb') as f:
        dill.dump(container, f)

if __name__ == '__main__':
    # generate_embed(model_name='bert')
    # generate_cls()
    save_dir = os.path.join(PARENT_DIR, 'models')
    logging.info(save_dir)
    main(save_at=save_dir, dataset=DATASET, light=True)