import argparse

import torch

from experiments import CONAN
from models import MORIRAN
from config import get_cfg
from trainer import Trainer
from datatool import load_dataset


dataset = {
    0: "Caltech101-20",
    1: "Scene-15",
    2: "LandUse-21",
    3: "NoisyMNIST",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-e', type=str, default='moriran', help='Experiment name.')
    # ['Scene-15', 'LandUse-21', 'Caltech101-20', 'NoisyMNIST']
    parser.add_argument('--dataset', '-d', type=int, default=0, help='Dataset id.')
    parser.add_argument('--ablation', action='store_true', help='Dataset id.')
    args = parser.parse_args()
    args.dataset = dataset[args.dataset]
    return args


if __name__ == '__main__':
    args = parse_args()
    config = get_cfg(args)
    device = torch.device(f"cuda:{config.device}")
    
    print(f'Use {device}')
    if config.experiment == 'moriran':
        model_cls = MORIRAN
    elif config.experiment == 'conan':
        model_cls = CONAN
    else:
        raise ValueError('model type error.')
    
    if config.task == 'clustering':
        train_dataset = load_dataset(config.train.dataset)
        trainer = Trainer(model_cls, config, train_dataset, valid_dataset=train_dataset, device=device)   
    else:
        train_dataset, valid_dataset, test_dataset = load_dataset(config.train.dataset, classification=True)
        trainer = Trainer(model_cls, config, train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, device=device)
    
    trainer.train()
    