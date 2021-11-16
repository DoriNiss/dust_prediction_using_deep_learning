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
data_dir = "../../data/datasets_20_81_189_averaged_dust_24h"
base_filename = "dataset_20_81_189_averaged_dust_24h"
description = torch.load(f"{data_dir}/metadata/dataset_20_81_189_averaged_dust_24h_metadata.pkl")
titles_channels = [description["input"][i]["short"] for i in range(20)]


inputs = torch.load(f"{data_dir}/{base_filename}_all_inputs.pkl")
targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
inputs.shape,targets.shape,len(timestamps)


inputs_events,targets_events,timestamps_events,idxs_events =     get_inputs_targets_timestamps_idxs_of_labels_only(inputs,targets,timestamps,1)

inputs_clear,targets_clear,timestamps_clear,idxs_clear =     get_inputs_targets_timestamps_idxs_of_labels_only(inputs,targets,timestamps,0)


seasons = ["DJF", "MAM", "JJA", "SON"]
timestamps_dict = {"clear":timestamps_clear, "events":timestamps_events}
seasons_months = {"DJF":[12,1,2], "MAM":[3,4,5], "JJA":[6,7,8], "SON":[9,10,11]}

seasonal_idxs = {"clear":{}, "events":{}}
seasonal_timestamps = {"clear":{}, "events":{}}
for time_type in ["clear", "events"]:
    for season in seasons:
        seasonal_idxs[time_type][season] = []
        seasonal_timestamps[time_type][season] = []

for time_type in ["clear", "events"]:
    for i,time in enumerate(timestamps_dict[time_type]):
        for season in ["DJF", "MAM", "JJA", "SON"]:
            if time.month in seasons_months[season]:
                seasonal_idxs[time_type][season]+=[i]
                seasonal_timestamps[time_type][season]+=[time]

for time_type in ["clear", "events"]:
    for season in seasons:
        seasonal_idxs[time_type][season] = np.array(seasonal_idxs[time_type][season])
        seasonal_timestamps[time_type][season] = pd.to_datetime(seasonal_idxs[time_type][season])
        print(f"\n\n#### {time_type}, {season}: length = {len(seasonal_idxs[time_type][season])}: "               f"\n{seasonal_timestamps[time_type][season][:5]}\n...\n{seasonal_timestamps[time_type][season][:5]}")


## Seasonal averages creation - saved to f"{data_dir}/{base_filename}_seasonal_averages.pkl"

# averages_seasonal = {"clear":{}, "events":{}, "global":{}}

# inputs_dict = {"clear":inputs_clear, "events":inputs_events}


# for time_type in ["clear", "events", "global"]:
#     print(f"#### {time_type}:")
#     for season in seasons:
#         if time_type == "global":
#             season_idxs_clear = seasonal_idxs["clear"][season]
#             season_idxs_events = seasonal_idxs["events"][season]
#             global_tensor = torch.cat([inputs_dict["clear"][season_idxs_clear],
#                                        inputs_dict["events"][season_idxs_events]])
#             old_shape = global_tensor.shape
#             averages_seasonal[time_type][season] = batch_average_datapoint(global_tensor)
#         else:
#             season_idxs = seasonal_idxs[time_type][season]
#             old_shape = inputs_dict[time_type][season_idxs].shape
#             averages_seasonal[time_type][season] = batch_average_datapoint(inputs_dict[time_type][season_idxs])
#         print(f"     {season} shape: {old_shape} -> {averages_seasonal[time_type][season].shape}")


# torch.save(averages_seasonal,f"{data_dir}/{base_filename}_seasonal_averages.pkl")
averages_seasonal = torch.load(f"{data_dir}/{base_filename}_seasonal_averages.pkl")


time_types = ["clear", "events", "global"]
tensors_seasonal_averages = [averages_seasonal[time_type][season] for time_type in time_types for season in seasons]
titles_seasonal_averages = [f"{time_type}, {season}" for time_type in time_types for season in seasons]
len(tensors_seasonal_averages), len(titles_seasonal_averages)


# tensors = [t[0] for t in tensors_seasonal_averages[:4]]*2
# titles = [t for t in titles_seasonal_averages[:4]]*2
# title = "Seasonal Averages"

# print_tensors_with_cartopy(tensors, main_title=title, titles=titles,
#                            lons=None, lats=None, save_as="", lock_bar=True, lock_bar_idxs=None,
#                            num_levels=10)    





for season in seasons:
    print(f"#### Season: {season}")
    for time_type in [""]
    daily_avgs_events,daily_avgs_clear = [],[]
    times_clear = timestamps_dict["clear"]
    times_events = timestamps_dict["events"]

    seasonal_idxs = {"clear":{}, "events":{}}
    for i in range(0,8):
        events_i,_,_,_ =             get_inputs_targets_timestamps_idxs_above_or_below_value(inputs,targets,timestamps,
                                                                    label_th=73.4,label_idx=i,
                                                                    above_or_below="above")
        clear_i,_,_,_ =             get_inputs_targets_timestamps_idxs_above_or_below_value(inputs,targets,timestamps,
                                                                    label_th=73.4,label_idx=i,
                                                                    above_or_below="below")
        daily_avgs_events.append(batch_average_datapoint(inputs_events_i))
        daily_avgs_clear.append(batch_average_datapoint(inputs_clear_i))
            
    for c in range(20):
        for i in tqdm(range(0,8)):
            inputs_season = 


        for i in tqdm(range(8)):
            inputs_averages_i,targets_averages_i,timestamps_averages_i,idxs_averages_i =                 get_inputs_targets_timestamps_idxs_above_or_below_value(inputs,targets,timestamps,
                                                                        label_th=10000,label_idx=i,
                                                                        above_or_below="below")
            averages_all.append(batch_average_datapoint(inputs_averages_i))
            titles_averages+=[f"Average all: in {i} days"]        title = titles_channels[c]
        tensors = 
        














tensors_averages = [inputs_events_avgs[i]-averages_all[i] for i in range(7,-1,-1)] +                    [inputs_clear_avgs[i]-averages_all[i] for i in range(7,-1,-1)] +                    [averages_all[i] for i in avgs_idxs]
                            
titles = [f"Average Event - Average: in {i} days" for i in range(7,-1,-1)]+          [f"Average Clear - Average: in {i} days" for i in range(7,-1,-1)]+          [f"Average: in {i} days" for i in avgs_idxs]

for c in range(20):
    title = f"Channel: {titles_channels[c]}"
    tensors = [t[c] for t in tensors_averages]
    print_tensors_with_cartopy(tensors, main_title=title, titles=titles,
                               lons=None, lats=None, save_as="", lock_bar=True, lock_bar_idxs=list(range(16)),
                               num_levels=10)    











inputs_events_avgs,inputs_clear_avgs = [],[]
averages_global = []
averages_seasonal = []
titles_averages = []
titles_global_averages = []

for i in tqdm(range(0,8)):
    inputs_events_i,targets_events_i,timestamps_events_i,idxs_events_i =         get_inputs_targets_timestamps_idxs_above_or_below_value(inputs,targets,timestamps,label_th=73.4,label_idx=i,
                                                                above_or_below="above")
    inputs_clear_i,targets_clear_i,timestamps_clear_i,idxs_clear_i =         get_inputs_targets_timestamps_idxs_above_or_below_value(inputs,targets,timestamps,label_th=73.4,label_idx=i,
                                                                above_or_below="below")
    inputs_events_avgs.append(batch_average_datapoint(inputs_events_i))
    inputs_clear_avgs.append(batch_average_datapoint(inputs_clear_i))


for i in tqdm(range(8)):
    inputs_averages_i,targets_averages_i,timestamps_averages_i,idxs_averages_i =         get_inputs_targets_timestamps_idxs_above_or_below_value(inputs,targets,timestamps,label_th=10000,label_idx=i,
                                                                above_or_below="below")
    averages_all.append(batch_average_datapoint(inputs_averages_i))
    titles_averages+=[f"Average all: in {i} days"]
    

