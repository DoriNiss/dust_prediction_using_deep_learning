import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate 
import numpy as np
import sys
sys.path.insert(0, '../../packages/')
from data_handlers.augmentations import *

"""
    To be used like that:
    Inside training/validation loop: 
        for minibatch, _ in loader:
    dataloader definition:
        train_dataset = DustPredictionDataset(....),
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=dust_prediction_collate)
    augmentation_tensor size: N,self.meteorology[1:] (one tensor of both clear and event days - not from validation set!)
    Available agumentations:
        NoAugmentation (default)
        PerlinAugmentatation 
        other - you can specify a new instance of agumentation from data_handlers.augmentations
"""

class DustPredictionDataset(Dataset):
    def __init__(self, meteorology_full_tensor, dust_full_tensor, times, augmentation=None, th=73.4, dust_idx=0, 
                 importance_events_ratio=0.4):
        self.meteorology = meteorology_full_tensor
        self.dust = dust_full_tensor
        self.times = times
        self.augmentation = augmentation or NoAugmentation()
        self.th = th
        self.labels = torch.where(dust_full_tensor[:,dust_idx]>=th, 
                                  dust_full_tensor.new_tensor([1.]), 
                                  dust_full_tensor.new_tensor([0.]))  # 1 are events, 0 are clear days
        self.default_importance_events_ratio = importance_events_ratio # if set to None - will return simply the batch without duplicating by importance
        self.dust_idx = dust_idx
        self.collate_fn = dust_prediction_collate
 
    def __len__(self):
        return self.dust.shape[0]

    def __getitem__(self, idx):
        # x = self.augmentation.augment(self.meteorology[idx,:,:,:], self.dust[idx,:])
        return self.meteorology[idx] , self.dust[idx], self.times[idx]
    
    def sample_by_importance(self, meteorology, dust, events_ratio=None, debug=False):
        """
            Input: meteorology - [batch_size,self.meteorology.shape[1:]]
                   dust - [batch_size,self.dust.shape[1:]]
            Output: (meteorology, dust), but with some clear days' rows replaced with duplications of dust event' rows.
            The number of replaced rows makes the ratio of events to batch_size equals to events_ratio
            If can't pupolate the correct amount of events - the output is regular (no importance sampling)
        """
        default_return = (meteorology, dust)  
        if events_ratio==-1:
            return default_return 
        r = events_ratio or self.default_importance_events_ratio
        batch_size = meteorology.shape[0]     
        final_num_events = int(r*batch_size)
        try:
            events_idxs = (dust[:,self.dust_idx]>=self.th).nonzero()[:,0]
            clear_idxs = (dust[:,self.dust_idx]<self.th).nonzero()[:,0]
            num_events_to_populate = final_num_events-events_idxs.shape[0]
            events_to_populate_from_idxs = events_idxs[torch.multinomial(events_idxs*1.,
                                                                         num_samples=num_events_to_populate,
                                                                         replacement=True)]
            clear_to_populate_idxs = clear_idxs[torch.multinomial(clear_idxs*1.,
                                                                  num_samples=num_events_to_populate,
                                                                  replacement=False)]
            meteorology_by_importance,dust_by_importance = meteorology,dust
            meteorology_by_importance[clear_to_populate_idxs] = meteorology_by_importance[events_to_populate_from_idxs]
            dust_by_importance[clear_to_populate_idxs] = dust_by_importance[events_to_populate_from_idxs]
            return meteorology_by_importance, dust_by_importance
        except Exception as exc:
            if debug:
                print("Could not perform importance sampling for this batch:")
                print(exc)
                print("dust:",dust[:,self.dust_idx],"events:",(dust[:,self.dust_idx]>=self.th).nonzero(),
                      (dust[:,self.dust_idx]>=self.th).nonzero().shape)
                print("final_num_events:",final_num_events,"num_events_to_populate = final_num_events-events_idxs.shape[0]")
            return default_return

def dust_prediction_collate(batch):
    new_batch = []
    timestamps = []
    for _batch in batch:
        new_batch.append(_batch[:-1])
        timestamps.append(_batch[-1])
    return default_collate(new_batch), timestamps


