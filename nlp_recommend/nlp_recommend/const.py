import os

CUR_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CUR_DIR)))
WEIGHT_DIR = os.path.join(PARENT_DIR, 'training')
PARENT_PARENT_DIR = os.path.dirname(PARENT_DIR)
DATA_PATH = os.path.join(WEIGHT_DIR, 'dataset/sentences.csv')
DATASET_PATH = os.path.join(WEIGHT_DIR, 'dataset') # location of sentences csv files
ORG_TXT_DIR = os.path.join(PARENT_DIR, 'data/gutenberg')
