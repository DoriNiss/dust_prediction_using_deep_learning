#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/')
from utils.files_loading import *
from data_handlers.tensors_labeling_and_averaging import *
from tqdm import tqdm


data_dir_original = "../../data/datasets_20_81_189"
data_dir_new = "../../data/datasets_20_81_189_averaged_dust_24h"
base_filename_original = "dataset_20_81_189"
base_filename_new = "dataset_20_81_189_averaged_dust_24h"
suffix_target = "target.pkl"
suffix_input = "input.pkl"
suffix_timestamps = "timestamps.pkl"
years_list = list(range(2003,2019))


thresholds = [73.4]
labels = [0,1]


for y in tqdm(years_list[0:1]):
    original_target = torch.load(f"{data_dir_original}/{base_filename_original}_{y}_{suffix_target}")
    original_target_labeled = add_labels(original_target,thresholds=thresholds,labels=labels,label_by_col=0)
    print(original_target_labeled[1030:1040,0],original_target_labeled[1030:1040,-1])
    
    

















import torch


a = torch.ones([5,8])
a[0,0]=0
a[0,5]=0
a[1,1]=0
a[1,2]=0
a[1,3]=0
a[1,4]=0
a[2,4]=0
a[2,6]=0
a[3,1]=0
a[3,2]=0
a[3,3]=0
a[4,5]=0
a[4,6]=0
a[4,7]=0

a


add_labels(a, thresholds=[0], labels=[0,1], label_by_col=0)


cols_to_average = [np.arange(4*i,4*(i+1)) for i in range(2)]
valid_threshold = [0.5]*2
cols_to_average,valid_threshold


x_averaged,rows_to_keep = average_cols_and_drop_invalid(a,cols_to_average,valid_threshold,invalid_values=[0])
print(x_averaged)
print(rows_to_keep)


# create new inputs and save
# create new metatdata and save


import numpy as np


a = np.arange(0,4)
len(a)







