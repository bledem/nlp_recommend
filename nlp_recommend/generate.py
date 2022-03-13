"""
Script to generate all pickles file for weights, sentiment labels 
from one dataset. It includes Model+Sentiment+Warper for a degree of accuracy/data weight. 
"""
from dataclasses import dataclass
import sys
import os

PARENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PARENT_DIR)

import dill 
from nlp_recommend.models.container import Container, ContainerLight, ContainerVeryLight

def main(save_at, dataset, weight_dir, dataset_path, light='no'):
    """_summary_

    Args:
        save_at (_type_): _description_
        dataset (_type_): _description_
        weight_dir (_type_): _description_
        dataset_path (_type_): folder containing *_clean.csv where sentences are located (for warper)
        light (str, optional): _description_. Defaults to 'no'.
    """
    if light=='no':
        container = ContainerLight(dataset, weight_dir=weight_dir, dataset_path=dataset_path)
        saving_at = os.path.join(save_at, f'{dataset}_container_light.pkl')
    elif light=='yes':
        container = Container(dataset, weight_dir=weight_dir, dataset_path=dataset_path)
        saving_at = os.path.join(save_at, f'{dataset}_container.pkl')
    elif light=='very':
        container = ContainerVeryLight(dataset, weight_dir=weight_dir, dataset_path=dataset_path)
        saving_at = os.path.join(save_at, f'{dataset}_container_verylight.pkl') 
    with open(saving_at, 'wb') as f:
        dill.dump(container, f)

if __name__ == '__main__':

    weight_dir = '/Users/10972/Documents/NLP_PJ/training'
    # weight_dir = '/app/training'
    dataset_path = os.path.join(weight_dir, 'dataset')
    save_dir = os.path.join(weight_dir, 'models')

    DATASET = 'philosophy'
    main(save_dir, DATASET, weight_dir, dataset_path, light='very')
    
    DATASET = 'adventure'
    main(save_dir, DATASET, weight_dir, dataset_path, light='very')
    
    DATASET = 'psychology'
    main(save_dir, DATASET, weight_dir, dataset_path, light='very')
