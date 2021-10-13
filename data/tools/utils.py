import os
import sys

from ..datasets import VisualGenomeTrainData

def register_datasets(cfg):
    if cfg.DATASETS.TYPE == 'VISUAL GENOME':
        for split in ['train', 'val', 'test']:
            dataset_instance = VisualGenomeTrainData(cfg, split=split)
        