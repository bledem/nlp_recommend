import os

CUR_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CUR_DIR)
DATA_PATH = os.path.join(PARENT_DIR, 'dataset/sentences.csv')
DATASET_PATH = os.path.join(PARENT_DIR, 'dataset')
ORG_TXT_DIR =  os.path.join(os.path.dirname(PARENT_DIR), 'data/gutenberg')
