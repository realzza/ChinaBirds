import json
from torch import DoubleTensor
from .sampler import TruncateBatchSampler
from .collate import PadCollate
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler


class SimpleDataLoader(DataLoader):
    def __init__(self, dataset,
                 shuffle=False, batch_size=1, num_workers=0, drop_last=False, pin_memory=True):
        self.dataset = dataset
        super().__init__(self.dataset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         pin_memory=pin_memory)
        


class WeightedDataLoader(DataLoader):
    def __init__(self, dataset, weight_path,
                 shuffle=False, batch_size=1, num_workers=0, drop_last=False, pin_memory=True):
        self.dataset = dataset
        with open(weight_path,'r') as f:
            self.sampler_weights = list(json.load(f).values())
        self.sampler_weights = DoubleTensor(self.sampler_weights)  
        self.sampler = WeightedRandomSampler(self.sampler_weights, len(self.sampler_weights))
        super().__init__(self.dataset,
                         sampler=self.sampler,
                         batch_size=batch_size,
                         num_workers=num_workers,
#                          shuffle=shuffle,
                         drop_last=drop_last,
                         pin_memory=pin_memory)



class TruncateDataLoader(DataLoader):
    def __init__(self, dataset, trun_range=[200, 400], step=1, 
                 shuffle=False, batch_size=1, num_workers=0, drop_last=False, pin_memory=True):
        self.dataset = dataset
        if shuffle:
            self.sampler = RandomSampler(self.dataset)
        else:
            self.sampler = SequentialSampler(self.dataset)
        self.batch_sampler = TruncateBatchSampler(
            self.sampler, batch_size, drop_last, trun_range, step)

        super().__init__(self.dataset, 
                         batch_sampler=self.batch_sampler, 
                         num_workers=num_workers, 
                         pin_memory=pin_memory)



class PadDataLoader(DataLoader):
    """
    Padding batch data to be as long as the longest sequence in the batch. 
    """
    def __init__(self, dataset, dim=1, mode='zeros', 
                 shuffle=False, batch_size=1, num_workers=0, drop_last=False, pin_memory=True):
        self.dataset = dataset
        self.pad_collate = PadCollate(dim, mode)
        super().__init__(self.dataset,
                         collate_fn=self.pad_collate, 
                         batch_size=batch_size,
                         num_workers=num_workers,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         pin_memory=pin_memory)

