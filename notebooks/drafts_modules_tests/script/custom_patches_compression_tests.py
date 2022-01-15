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


patch_idxs_rows = [np.array([2]),np.arange(2),np.arange(6)]
patch_idxs_cols = [np.arange(7), np.arange(3),np.arange(9,21)]
patch_sizes =     [[27,27],       [27,27],    [9,9]]
week_idxs = [5,4,3,2,1,0]


years_list = list(range(2003,2019))
data_dir = "../../data/datasets_20_81_189_3h_7days_future"
base_filename = "dataset_20_81_189_3h_7days_future"
description = torch.load(f"{data_dir}/metadata/dataset_20_81_189_3h_7days_future_2003_descriptions.pkl")[0]
description_reduced_channels_path = f"{data_dir}/metadata/dataset_20_81_189_3h_7days_future_all_reduced_channels_description.pkl"
description_reduced_channels = torch.load(description_reduced_channels_path)
titles_channels_all = [description["input"][i]["short"] for i in range(20)]
titles_channels_all_long = [description["input"][i]["long"] for i in range(20)]
titles_channels_reduced = [description_reduced_channels["input"][i]["long"] for i in range(5)]
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


a = torch.ones([1,1,81,189])
a[:,:,3,:] = 10
a[:,:,-1,:] = 20
a[:,:,:,2] = 15


a_patched,a_values = calculate_patches_and_values(a,patch_idxs_rows,patch_idxs_cols,patch_sizes)
a_patched.shape,a_values.shape


plt.imshow(a[0,0])


print("Means")
plt.imshow(a_patched[0,0,0])
plt.show();
print("Mins")
plt.imshow(a_patched[1,0,0])
plt.show();
print("Maxs")
plt.imshow(a_patched[2,0,0])
plt.show();
print("STDs")
plt.imshow(a_patched[3,0,0])
plt.show();


plt.imshow(a_patched[1,0,0])


a_values[0,0,:,2]








t = inputs_debug[0:1,0:1,:,:]
t.shape


t_patched,t_values = calculate_patches_and_values(t,patch_idxs_rows,patch_idxs_cols,patch_sizes)
t_patched.shape,t_values.shape


plt.imshow(t[0,0])


print("Means")
plt.imshow(t_patched[0,0,0])
plt.show();
print("Mins")
plt.imshow(t_patched[1,0,0])
plt.show();
print("Maxs")
plt.imshow(t_patched[2,0,0])
plt.show();
print("STDs")
plt.imshow(t_patched[3,0,0])
plt.show();


cols_titles_full = ["Raw","Means","Mins","Maxs","STDs","Skews","Kurtosis"]


tensors = [a[0,0]]+[a_patched[i,0,0] for i in range(3)]+           [t[0,0]]+[t_patched[i,0,0] for i in range(3)]

rows_titles = ["a","t"]
cols_titles = [cols_titles_full[i] for i in range(4)]
print_tensors_with_cartopy(tensors, main_title="", titles=None, lock_bar=False, 
                       lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=4,
                       levels_around_zero=False,titles_only_on_edges=True,lock_bar_rows_separately=True,
                       cols_titles=cols_titles, rows_titles=rows_titles)







