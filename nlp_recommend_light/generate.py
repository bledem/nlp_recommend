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
from nlp_recommend.models.container import Container, ContainerLight, ContainerVeryLight

def main(save_at, dataset, weight_dir, dataset_path, light='no'):
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
    dataset_path = os.path.join(weight_dir, 'dataset')
    save_dir = os.path.join(weight_dir, 'models')

    DATASET = 'philosophy'
    main(save_dir, DATASET, weight_dir, dataset_path, light='very')
    
    DATASET = 'adventure'
    main(save_dir, DATASET, weight_dir, dataset_path, light='very')
    
    DATASET = 'psychology'
    main(save_dir, DATASET, weight_dir, dataset_path, light='very')
