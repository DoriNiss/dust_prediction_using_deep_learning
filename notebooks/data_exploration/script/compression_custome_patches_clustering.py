#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/')
from utils.files_loading import *
from utils.data_exploration import *
from utils.meteorology_printing import *

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import matplotlib.pyplot as plt


years_list = list(range(2003,2019))
data_dir = "../../data/datasets_20_81_189_3h_7days_future"
base_filename = "dataset_20_81_189_3h_7days_future"
description = torch.load(f"{data_dir}/metadata/dataset_20_81_189_3h_7days_future_2003_descriptions.pkl")[0]
titles_channels = [description["input"][i]["short"] for i in range(20)]
titles_channels_long = [description["input"][i]["long"] for i in range(20)]
event_threshold = 73.4

sequences_names = [
    "4days_light",
    "6days_light",
    "4days_heavy",
    "6days_heavy",
]


debug_year = 2005
inputs_debug = torch.load(f"{data_dir}/{base_filename}_{debug_year}_input.pkl")
targets_debug = torch.load(f"{data_dir}/{base_filename}_{debug_year}_target.pkl")
timestamps_debug = torch.load(f"{data_dir}/{base_filename}_{debug_year}_timestamps.pkl")
inputs_debug.shape,targets_debug.shape,len(timestamps_debug)


# inputs = torch.load(f"{data_dir}/{base_filename}_all_inputs.pkl")
# targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
# timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
# inputs.shape,targets.shape,len(timestamps)


t_sample = inputs_debug[5]
t_sample.shape


channels = [0,1,4,5,8,9,13,14]
tensors = [t_sample[c] for c in channels]
titles = [titles_channels[c] for c in channels]
print_tensors_with_cartopy(tensors, main_title="", titles=titles, num_rows=None, num_cols=None,
                            lons=None, lats=None, save_as="",lock_bar=False, lock_bar_idxs=None, 
                            num_levels=None, levels_around_zero=False, manual_levels=None)


t_patched = patch_averages(t_sample.unsqueeze(0),[27,27])
t_patched.shape


def print_channel_samples(tensors,main_title="",titles=None,channels=None,num_rows=None,num_cols=None,lons=None, 
                          lats=None, save_as="",lock_bar=False,lock_bar_idxs=None,num_levels=10,
                          levels_around_zero=False, manual_levels=None):
    channels = channels or [0,1,4,5,8,9,13,14]
    tensors = tensors if len(tensors)>=8 else [tensors[0][c] for c in channels] # assuming shape [C,H,W]
    titles = titles or [titles_channels[c] for c in channels]
    print_tensors_with_cartopy(tensors, main_title=main_title,titles=titles,num_rows=num_rows,num_cols=num_cols,
                               lons=lons,lats=lats,save_as=save_as,lock_bar=lock_bar,lock_bar_idxs=lock_bar_idxs, 
                               num_levels=num_levels,levels_around_zero=levels_around_zero,manual_levels=manual_levels)


t_patched_scaled = torch.nn.functional.interpolate(t_patched, size=[81,189], scale_factor=None, mode='nearest', align_corners=None)
t_patched_scaled.shape


print_channel_samples([t_patched_scaled[0]])


def calculate_patches_regions(t,idxs_h,idxs_w,patch_sizes):
    """
        t shape: [N,C,H,W]
        idxs_h,idxs_w,patch_sizes are lists of the same lengths
        idxs_h,idxs_w are lists of tuples that slice objects will be created based on, e.g. [(0,2),(4,10)]...
        idxs_h,idxs_w are lists of slice objects, e.g. [slice(-2:0:-1),slice(4,10)]...
    """
    out = torch.zeros_like(t)
    H,W = t.shape[-2],t.shape[-1]
    for i,patch_size in enumerate(patch_sizes):
#         if len(idxs_h[i])==3:
# #             hs = slice(int(idxs_h[i][0]),int(idxs_h[i][1]),int(idxs_h[i][2]))
#             hs = slice(idxs_h[i][0]:idxs_h[i][1]:idxs_h[i][2])
#         elif len(idxs_h[i])==2:
# #             hs = slice(int(idxs_h[i][0]),int(idxs_h[i][1]))
#             hs = slice(idxs_h[i][0]:idxs_h[i][1])
#         else:
#             print(f"Error! Bad length of slice h: {idxs_h[i]} at i={i}")
#         if len(idxs_w[i])==3:
#             ws = slice(int(idxs_w[i][0]),int(idxs_w[i][1]),int(idxs_w[i][2]))
#         elif len(idxs_w[i])==2:
#             ws = slice(int(idxs_w[i][0]),int(idxs_w[i][1]))
#         else:
#             print(f"Error! Bad length of slice w: {idxs_w[i]} at i={i}")
        hs,ws = idxs_h[i],idxs_w[i]
        t_patched = patch_averages(t,patch_size)
        t_patched = torch.nn.functional.interpolate(t_patched,size=[H,W])
        out[:,:,hs,ws]+=t_patched[:,:,hs,ws]
    return out
        


idxs_h =      [slice(27*2,27*3),slice(0,27*2),slice(0,27*2)]
idxs_w =      [slice(0,27*7),   slice(0,27*3),slice(27*3,27*7)]
patch_sizes = [[27,27],         [27,27],      [9,9]]
t_custome_patches = calculate_patches_regions(t_sample.unsqueeze(0),idxs_h,idxs_w,patch_sizes)
t_custome_patches.shape


print_channel_samples([t_custome_patches[0]])


### 1) Try to average all events like that. 2) Get one value from each patch
### Try the same for max values

