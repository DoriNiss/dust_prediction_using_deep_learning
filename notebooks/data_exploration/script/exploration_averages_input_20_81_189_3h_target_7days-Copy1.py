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


inputs = torch.load(f"{data_dir}/{base_filename}_all_inputs.pkl")
targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
inputs.shape,targets.shape,len(timestamps)





# clear_avg = inputs_clear.mean(0)
# inputs_events_minus_clear_avgs.shape


description["target"]


description["input"]


cols = [7,6,5,4,3,2,1,0]

tensors_all_channels = []
titles = []
title_base = "Average Events Anonamlies Progression:"

inputs_avgs = inputs.mean(0)

for col in tqdm(cols):
    idxs = targets[:,col]>=event_threshold
    inputs_events_minus_avgs = inputs[idxs].mean(0)-inputs_avgs
    tensors_all_channels.append(inputs_events_minus_avgs)
    titles.append(f"Event in {col} days")
    print(f"{titles[-1]}: {idxs.sum()}")


# # for c in range(20):
# levels = {}

# def calc_levels(vmin,vmax,num_steps):
#     step_size = (vmax-vmin)/num_steps
#     return np.arange(vmin,vmax+step_size/2,step=step_size)

# levels[10] = calc_levels(-0.5,1.55,10)
# levels[11] = calc_levels(-0.5,1.55,10)
# levels[12] = calc_levels(-0.5,1.55,10)
# levels[13] = calc_levels(-0.5,1.55,10)

# levels[14] = calc_levels(-0.12,0.2,10)
# levels[15] = calc_levels(-0.12,0.16,10)
# levels[16] = calc_levels(-9e-5,16e-5,10)
# levels[17] = calc_levels(-4.5e-5,6e-5,10)


for c in range(20):
    print(f"Channel: {titles_channels_long[c]}")
    levels_around_zero = c<10 or c>=18
    levels_plot = None #if c<10 else levels[c]
    title = f"{title_base} {titles_channels_long[c]}"
    tensors = [t[c] for t in tensors_all_channels]
    print_tensors_with_cartopy(tensors, main_title=title, titles=titles, lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=levels_plot,
                               levels_around_zero=levels_around_zero)   


























# SEASONAL - NOT GOOD YET...


inputs_events,targets_events,timestamps_events,idxs_events =     get_inputs_targets_timestamps_idxs_of_labels_only(inputs,targets,timestamps,1)

inputs_clear,targets_clear,timestamps_clear,idxs_clear =     get_inputs_targets_timestamps_idxs_of_labels_only(inputs,targets,timestamps,0)


seasons = ["DJF", "MAM", "JJA", "SON"]
timestamps_dict = {"clear":timestamps_clear, "events":timestamps_events, "global":timestamps}
seasons_months = {"DJF":[12,1,2], "MAM":[3,4,5], "JJA":[6,7,8], "SON":[9,10,11]}
time_types = ["clear","events","global"]

seasonal_idxs = {t_type:{} for t_type in time_types}
seasonal_timestamps = {t_type:{} for t_type in time_types}
for time_type in time_types:
    for season in seasons:
        seasonal_idxs[time_type][season] = []
        seasonal_timestamps[time_type][season] = []

for time_type in time_types:
    for i,time in enumerate(timestamps_dict[time_type]):
        for season in ["DJF", "MAM", "JJA", "SON"]:
            if time.month in seasons_months[season]:
                seasonal_idxs[time_type][season]+=[i]
                seasonal_timestamps[time_type][season]+=[time]

for time_type in time_types:
    for season in seasons:
        seasonal_idxs[time_type][season] = np.array(seasonal_idxs[time_type][season])
        seasonal_timestamps[time_type][season] = pd.to_datetime(seasonal_timestamps[time_type][season])
        print(f"\n\n#### {time_type}, {season}: length = {len(seasonal_idxs[time_type][season])}: "               f"\n{seasonal_timestamps[time_type][season][:5]}\n...\n{seasonal_timestamps[time_type][season][-5:]}")


# ## Seasonal averages creation - saved to f"{data_dir}/{base_filename}_seasonal_averages.pkl"

# averages_seasonal = {t_type:{} for t_type in time_types}

# inputs_dict = {"clear":inputs_clear, "events":inputs_events, "global": inputs}


# for time_type in time_types:
#     print(f"#### {time_type}:")
#     for season in seasons:
#         season_idxs = seasonal_idxs[time_type][season]
#         old_shape = inputs_dict[time_type][season_idxs].shape
#         averages_seasonal[time_type][season] = batch_average_datapoint(inputs_dict[time_type][season_idxs])
#         print(f"     {season} shape: {old_shape} -> {averages_seasonal[time_type][season].shape}")


# # # #### clear:
# # #      DJF shape: torch.Size([3031, 20, 81, 189]) -> torch.Size([20, 81, 189])
# # #      MAM shape: torch.Size([3076, 20, 81, 189]) -> torch.Size([20, 81, 189])
# # #      JJA shape: torch.Size([3943, 20, 81, 189]) -> torch.Size([20, 81, 189])
# # #      SON shape: torch.Size([3165, 20, 81, 189]) -> torch.Size([20, 81, 189])
# # # #### events:
# # #      DJF shape: torch.Size([619, 20, 81, 189]) -> torch.Size([20, 81, 189])
# # #      MAM shape: torch.Size([711, 20, 81, 189]) -> torch.Size([20, 81, 189])
# # #      JJA shape: torch.Size([176, 20, 81, 189]) -> torch.Size([20, 81, 189])
# # #      SON shape: torch.Size([414, 20, 81, 189]) -> torch.Size([20, 81, 189])
# # # #### global:
# # #      DJF shape: torch.Size([3650, 20, 81, 189]) -> torch.Size([20, 81, 189])
# # #      MAM shape: torch.Size([3787, 20, 81, 189]) -> torch.Size([20, 81, 189])
# # #      JJA shape: torch.Size([4119, 20, 81, 189]) -> torch.Size([20, 81, 189])
# # #      SON shape: torch.Size([3579, 20, 81, 189]) -> torch.Size([20, 81, 189])


# torch.save(averages_seasonal,f"{data_dir}/{base_filename}_seasonal_averages.pkl")
averages_seasonal = torch.load(f"{data_dir}/{base_filename}_seasonal_averages.pkl")


# inputs_minus_seasonal = inputs*0.
# for season in seasons:
#     idxs = seasonal_idxs["global"][season]
#     inputs_minus_seasonal[idxs] = inputs[idxs]-averages_seasonal["global"][season]
    





torch.save(inputs_minus_seasonal,f"{data_dir}/{base_filename}_inputs_minus_seasonal.pkl")
# inputs_minus_seasonal = torch.load(f"{data_dir}/{base_filename}_inputs_minus_seasonal.pkl")


inputs_minus_seasonal.shape


f"{data_dir}/{base_filename}_inputs_minus_seasonal.pkl"


# for season in seasons:
#     print(f"#### Season: {season}")
#     events_week,clear_week,global_week = [],[],[]
#     for i in range(0,8):
#         season_idxs = seasonal_idxs["events"][season]
#         events_i,_,_,_ = \
#             get_inputs_targets_timestamps_idxs_above_or_below_value(inputs_minus_seasonal[season_idxs],
#                                                                     targets[season_idxs],timestamps[season_idxs],
#                                                                     label_th=event_threshold,label_idx=i,
#                                                                     above_or_below="above")
#         season_idxs = seasonal_idxs["clear"][season]
#         clear_i,_,_,_ = \
#             get_inputs_targets_timestamps_idxs_above_or_below_value(inputs_minus_seasonal[season_idxs],
#                                                                     targets[season_idxs],timestamps[season_idxs],
#                                                                     label_th=event_threshold,label_idx=i,
#                                                                     above_or_below="below")
#         season_idxs = seasonal_idxs["global"][season]
#         global_i,_,_,_ = \
#             get_inputs_targets_timestamps_idxs_above_or_below_value(inputs_minus_seasonal[season_idxs],
#                                                                     targets[season_idxs],timestamps[season_idxs],
#                                                                     label_th=100000,label_idx=i,
#                                                                     above_or_below="below")
#         events_week.append(batch_average_datapoint(events_i))
#         clear_week.append(batch_average_datapoint(clear_i))
#         global_week.append(batch_average_datapoint(global_i))
            
#     avgs_idxs = [6,4,2,0]
#     tensors_week = [events_week[i] for i in range(7,-1,-1)] + \
#                    [clear_week[i] for i in range(7,-1,-1)] + \
#                    [global_week[i] for i in avgs_idxs]

#     titles = [f"Average Event-Seasonal: Event in {i} days" for i in range(7,-1,-1)]+ \
#              [f"Average Clear-Seasonal: Clear in {i} days" for i in range(7,-1,-1)]+ \
#              [f"Seasonal Average: {i} days" for i in avgs_idxs]

#     for c in range(20):
#         title = f"Channel: {titles_channels[c]}"
#         tensors = [t[c] for t in tensors_week]
#         print_tensors_with_cartopy(tensors, main_title=title, titles=titles,
#                                    lons=None, lats=None, save_as="", lock_bar=True, lock_bar_idxs=list(range(16)),
#                                    num_levels=10)    














tensors_seasonal_averages = [averages_seasonal[time_type][season] 
                             for time_type in time_types for season in seasons]
titles_seasonal_averages = [f"{time_type}, {season}" for time_type in time_types for season in seasons]
len(tensors_seasonal_averages), len(titles_seasonal_averages)





inputs_season_reduced = inputs
for t in timestamps:
    idxs_season = 
    inputs_season_reduced[timestamps]





# tensors = [t[0] for t in tensors_seasonal_averages[:4]]*2
# titles = [t for t in titles_seasonal_averages[:4]]*2
# title = "Seasonal Averages"

# print_tensors_with_cartopy(tensors, main_title=title, titles=titles,
#                            lons=None, lats=None, save_as="", lock_bar=True, lock_bar_idxs=None,
#                            num_levels=10)    






        














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
    

