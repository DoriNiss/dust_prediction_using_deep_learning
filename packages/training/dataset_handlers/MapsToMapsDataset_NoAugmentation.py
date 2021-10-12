# +
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate 
import numpy as np
import sys
import pandas as pd
# sys.path.insert(0, '../../../packages/training/')
# from data_handlers.augmentations import *


class MapsToMapsDataset_NoAugmentation(Dataset):
    def __init__(self, input_tensor, target_tensor, timestamps):
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor
        self.timestamps = timestamps
        self.timestamps_as_ints = pd.to_numeric(self.timestamps)
 
    def __len__(self):
        return self.input_tensor.shape[0]

    def __getitem__(self, idx):
        return self.input_tensor[idx] , self.target_tensor[idx], self.timestamps_as_ints[idx]
    
    def get_date_from_int_timestamp(self, int_date):
        return pd.to_datetime(int_date)
    
