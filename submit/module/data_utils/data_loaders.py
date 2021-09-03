from torch.utils.data import DataLoader
from .datasets import MaskDataset
import numpy as np
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler, WeightedRandomSampler
from catalyst.data.sampler import BalanceClassSampler


class MaskDataLoader(DataLoader):
    def __init__(self, dataset, batch_size,  num_workers=1, shuffle=False, sampler=None, collate_fn=None):
        self.sampler = sampler
        self.shuffle = shuffle
        self.targets = dataset.labels
        self.class_cnt = np.unique(self.targets, return_counts=True)[1]

        if self.sampler == 'WeightedRandomSampler':
            self.sampler = self.weighted_sampler()
        elif self.sampler == 'BalanceClassSampler':
            self.sampler = BalanceClassSampler(
                labels=self.targets.values, mode=100)
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def weighted_sampler(self):
        weight = 1. / self.class_cnt
        samples_weight = weight[self.targets]
        samples_weight = torch.from_numpy(samples_weight)
        return WeightedRandomSampler(samples_weight, len(samples_weight))
