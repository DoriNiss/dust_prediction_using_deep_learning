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


inputs = torch.load(f"{data_dir}/{base_filename}_all_inputs.pkl")
targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
inputs.shape,targets.shape,len(timestamps)





# clear_avg = inputs_clear.mean(0)
# inputs_events_minus_clear_avgs.shape


# description["target"]


# description["input"]


cols = [7,6,5,4,3,2,1,0]

tensors_all_channels_minus_avgs = []
tensors_all_channels_normalized = []
titles = []
titles_normalized = []
title_base = "Average Events Anonamlies Progression:"

inputs_avgs = inputs.mean(0)
inputs_stds = inputs.std(0)

for col in tqdm(cols):
    idxs = targets[:,col]>=event_threshold
    inputs_minus_avgs = inputs[idxs].mean(0)-inputs_avgs
    inputs_normalized = (inputs[idxs].mean(0)-inputs_avgs)/inputs_stds
    tensors_all_channels_minus_avgs.append(inputs_minus_avgs)
    tensors_all_channels_normalized.append(inputs_normalized)
    titles.append(f"Event in {col} days")
    titles_normalized.append(f"Event in {col} days, normalized")
    print(f"{titles[-1]}: {idxs.sum()}")


for c in range(20):
    print(f"Channel: {titles_channels_long[c]}")
    levels_around_zero = c<10 or c>=18
    levels_plot = None #if c<10 else levels[c]
    title = f"{title_base} {titles_channels_long[c]}"
    tensors = [t[c] for t in tensors_all_channels_minus_avgs]
    print_tensors_with_cartopy(tensors, main_title=title, titles=titles, lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=levels_plot,
                               levels_around_zero=levels_around_zero)
    title = f"{title_base} {titles_channels_long[c]}"
    tensors = [t[c] for t in tensors_all_channels_normalized]
    print_tensors_with_cartopy(tensors, main_title=title, titles=titles_normalized, lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=levels_plot,
                               levels_around_zero=levels_around_zero)

















means = inputs.mean(0)
stds = inputs.std(0)


inputs_normalized_per_channel_per_pixel = (inputs-means[None,:,:,:])/stds[None,:,:,:]


inputs_normalized_per_channel_per_pixel.shape


torch.save(inputs_normalized_per_channel_per_pixel,f"{data_dir}/{base_filename}_all_inputs_pixel_normalized.pkl")














channels_to_average = [np.arange(4),
                       np.array([4,5,6,18]),np.array([7,8,9,19]),
                       np.arange(10,14),np.arange(14,18)]

print(len(channels_to_average),channels_to_average)
for c in range(len(channels_to_average)):
    print(f"{c}:\n   {[description['input'][i]['short'] for i in channels_to_average[c]]}")


inputs_normalized_reduced_channels = average_related_channels(inputs_normalized_per_channel_per_pixel,channels_to_average)
inputs_normalized_reduced_channels.shape


titles = [f"Event in {col} days" for col in cols]
titles,len(titles)


cols = [7,6,5,4,3,2,1,0]

tensors_reduced_channels = []
title_base = "Average Events Anonamlies Progression:"

for col in tqdm(cols):
    idxs = targets[:,col]>=event_threshold
    tensors_reduced_channels.append(inputs_normalized_reduced_channels[idxs].mean(0))
    print(f"{titles[7-col]}: {idxs.sum()}")


print([t.shape for t in tensors_reduced_channels],len(tensors_reduced_channels))


titles_reduced_channels = [
    "Averaged Geopotential Height",
    "Averaged Northward Wind (U)",
    "Averaged Eastward Wind (V)",
    "Averaged Potential Vorticity (PV)",
    "Averaged AOD",
]

for c in range(5):
    print(f"Channel: {titles_reduced_channels[c]}")
    levels_around_zero = True #c<3
    levels_plot = None 
    title = f"{title_base} {titles_reduced_channels[c]}"
    tensors = [t[c] for t in tensors_reduced_channels]
    print_tensors_with_cartopy(tensors, main_title=title, titles=titles, lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=levels_plot,
                               levels_around_zero=levels_around_zero)











cols = [7,6,5,4,3,2,1,0]

tensors_all_channels_normalized = []
title_base = "Average Events Anonamlies Progression:"
for col in tqdm(cols):
    idxs = targets[:,col]>=event_threshold
    tensors_all_channels_normalized.append(inputs_normalized_per_channel_per_pixel[idxs].mean(0))
    print(f"{titles[-1]}: {idxs.sum()}, {tensors_all_channels_normalized[-1].shape}")


for c in range(20):
    print(f"Channel: {titles_channels_long[c]}")
    levels_around_zero = True #c<10 or c>=18
    levels_plot = None 
    title = f"{title_base} {titles_channels_long[c]}"
    tensors = [t[c] for t in tensors_all_channels_normalized]
    print(tensors[0].shape)
    print_tensors_with_cartopy(tensors, main_title=title, titles=titles, lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=levels_plot,
                               levels_around_zero=levels_around_zero)


torch.save(inputs_normalized_reduced_channels,f"{data_dir}/{base_filename}_all_inputs_pixel_normalized_reduced_channels.pkl")


description_reduced_channels = description.copy()
description_reduced_channels["input"] = {
    0: {"short": "avg(SLP,Z@850,500,250)","long":"Averaged Geopotential Height"},
    1: {"short": "avg(U@850,500,250,u10m)","long":"Averaged Northward Wind (U)"},
    1: {"short": "avg(V@850,500,250,v10m)","long":"Averaged Eastward Wind (V)"},
    3: {"short": "avg(PV@325,330,335,340)","long":"Averaged Potential Vorticity (PV)"},
    4: {"short": "avg(aod550,duaod550,aermssdul,aermssdum)","long":"Averaged AOD"},
}
description_reduced_channels_path = f"{data_dir}/metadata/dataset_20_81_189_3h_7days_future_all_reduced_channels_description.pkl"
torch.save(description_reduced_channels,description_reduced_channels_path)

















event_sequences = torch.load(f"{data_dir}/{base_filename}_events_sequences_{sequences_names[0]}_inputs.pkl")
event_sequences.shape


events_avgs_sequences = event_sequences.mean([0])





title_base = "Average Events Anonamlies Progression (sequences):"

for c in range(20):
    print(f"Channel: {titles_channels_long[c]}")
    levels_around_zero = c<10 or c>=18
    levels_plot = None #if c<10 else levels[c]
    title = f"{title_base} {titles_channels_long[c]}"
    tensors = [t for t in events_avgs_sequences[:,c,:,:]]*2
    print_tensors_with_cartopy(tensors, main_title=title, titles=titles[:10], lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=levels_plot,
                               levels_around_zero=levels_around_zero, num_cols=5)
















