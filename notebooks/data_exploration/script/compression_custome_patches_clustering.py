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
description_reduced_channels_path = f"{data_dir}/metadata/dataset_20_81_189_3h_7days_future_all_reduced_channels_description.pkl"
description_reduced_channels = torch.load(description_reduced_channels_path)
titles_channels_all = [description["input"][i]["short"] for i in range(20)]
titles_channels_all_long = [description["input"][i]["long"] for i in range(20)]
titles_channels_reduced = [description_reduced_channels["input"][i]["long"] for i in range(5)]
titles_channels_reduced_short = [description_reduced_channels["input"][i]["short"] for i in range(5)]
event_threshold = 73.4

sequences_names = [
    "4days_light",
    "6days_light",
    "4days_heavy",
    "6days_heavy",
]





# debug_year = 2005
# inputs_debug = torch.load(f"{data_dir}/{base_filename}_{debug_year}_input.pkl")
# targets_debug = torch.load(f"{data_dir}/{base_filename}_{debug_year}_target.pkl")
# timestamps_debug = torch.load(f"{data_dir}/{base_filename}_{debug_year}_timestamps.pkl")
# inputs_debug.shape,targets_debug.shape,len(timestamps_debug)


inputs_filename = "dataset_20_81_189_3h_7days_future_all_inputs_pixel_normalized_reduced_channels.pkl"
inputs = torch.load(f"{data_dir}/{inputs_filename}")
targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
inputs.shape,targets.shape,len(timestamps)


# !ls ../../data/datasets_20_81_189_3h_7days_future/metadata


events_mask = targets[:,0]>=event_threshold
events_idxs = np.arange(events_mask.shape[0])[events_mask]
events_raw = inputs[events_mask]
events_targets = targets[events_mask]
events_timestamps = timestamps[events_idxs]
events_raw.shape,events_targets.shape,len(events_timestamps)


# description_reduced_channels["target"]


patch_idxs_rows = [np.array([2]),np.arange(2),np.arange(6)]
patch_idxs_cols = [np.arange(7), np.arange(3),np.arange(9,21)]
patch_sizes =     [[27,27],       [27,27],    [9,9]]
week_idxs = [5,4,3,2,1,0]


events_patched_week = []
events_raw_week = []

print("Calculating events averages...")
for day in tqdm(week_idxs):
    print(f"### Days before: {day}")
    events_days_before = inputs[targets[:,day]>=event_threshold]
    events_raw_week.append(events_days_before.mean(0))
    print(f"    Raw shape: {events_days_before.shape} -> {events_raw_week[-1].shape}, compressing...")
    events_patched,_ = calculate_patches_and_values(events_days_before,patch_idxs_rows,patch_idxs_cols,patch_sizes)
    events_patched_week.append(events_patched.mean(1))
    print(f"    Compressed-reshaped shape: {events_patched.shape} -> {events_patched_week[-1].shape}")





# num_rows = 7
# num_cols = 6
# cols_titles = [f"{day} Days Earlier" for day in range(num_cols)]
# rows_titles = [v_title for v_title in values_titles]
# # cols_titles = [f"{i}" for i in range(num_cols)]
# # rows_titles = [f"{i*10}" for i in range(num_rows)]
# tensors = [events_patched_week[0][0,0,:,:]]*num_rows*num_cols
# print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
#                        lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=num_cols,
#                        levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=True,
#                        cols_titles=cols_titles, rows_titles=rows_titles)





values_titles = ["Patch Means","Patch Min Vals","Patch Max Vals","Patch STDs","Patch Skews","Patch Kurtosis","Raw Image"]
# events_raw_week

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}"
    cols_titles = [f"{day} Days Earlier" for day in week_idxs]
    rows_titles_all = [v_title for v_title in values_titles]
    tensors_all = [events_patched_week[day][v_idx,c,:,:] for v_idx in range(6) for day in range(6)] +                   [events_raw_week[day][c,:,:] for day in range(6)]
#     tensors = [t for t in tensors_all[:18]]+[t for t in tensors_all[-6:]]
#     rows_titles = [r for r in rows_titles_all[:3]]+[rows_titles_all[-1]]
#     print_tensors_with_cartopy(tensors, 
#                                main_title=title_main, titles=None, lock_bar=True, 
#                                lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=6,
#                                levels_around_zero=False,titles_only_on_edges=True,lock_bar_rows_separately=False,
#                                cols_titles=cols_titles, rows_titles=rows_titles)
#     tensors = tensors[18:-6]
#     rows_titles = rows_titles[3:-1]
    print_tensors_with_cartopy(tensors_all, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=6,
                               levels_around_zero=False,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles_all)











sequence_base_filenames = [f"{data_dir}/{base_filename}_events_sequences_normalized_reduced_{sequences_names[i]}_"
                          for i in range(len(sequences_names))]
sequence_compressed_base_filenames = [f"{data_dir}/compressed/"                                       f"{base_filename}_events_sequences_normalized_reduced_patch_compressed_1_{sequences_names[i]}_"
                                      for i in range(len(sequences_names))]


# patch compressed 1:
patch_idxs_rows = [np.array([2]),np.arange(2),np.arange(6)]
patch_idxs_cols = [np.arange(7), np.arange(3),np.arange(9,21)]
patch_sizes =     [[27,27],       [27,27],    [9,9]]


for seq_i,seq_path in enumerate(tqdm(sequence_base_filenames)):
    events_seq = torch.load(seq_path+"inputs.pkl")
    num_events,seq_len,C,H,W = events_seq.shape
    print(f"### Loaded sequence {seq_i}: {events_seq.shape}. Compressing...")
    _,compressed_seq = calculate_patches_and_values(events_seq.reshape(num_events*seq_len,C,H,W),
                                                    patch_idxs_rows,patch_idxs_cols,patch_sizes)
    _,_,channel_compressed_size,num_values = compressed_seq.shape
    compressed_seq = compressed_seq.reshape(num_events,seq_len*C*channel_compressed_size,num_values)
    torch.save(compressed_seq,sequence_compressed_base_filenames[seq_i]+"inputs.pkl")
    print(f"    Done and saved! shape: {compressed_seq.shape}")
    














# debugging compressions


events_sequences_avgs = []
events_sequences = []
for seq_i,seq_path in enumerate(sequence_base_filenames[:-1]):
    events_seq = torch.load(seq_path+"inputs.pkl")
    events_sequences.append(events_seq)
    events_sequences_avgs.append(events_seq.mean(0))
    print(f"Loaded sequence #{seq_i} of shape {events_seq.shape} -> {events_sequences_avgs[-1].shape}")





titles_channels_reduced_short


i = 0
title_main = f"Events Averaged Progression, sequence #{i}: {sequences_names[i]}"
tensors = [events_sequences_avgs[i][day,c,:,:] for c in range(5) for day in range(5)]
cols_titles = [f"{day-1} Days Earlier" for day in week_idxs]
rows_titles = [row for row in titles_channels_reduced_short]
print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                           lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=5,
                           levels_around_zero=False,titles_only_on_edges=True,lock_bar_rows_separately=True,
                           cols_titles=cols_titles, rows_titles=rows_titles)


tesnors_patched,tensors_values = calculate_patches_and_values(events_sequences_avgs[i],
                                                              patch_idxs_rows,patch_idxs_cols,patch_sizes)
tesnors_patched.shape,tensors_values.shape


values_titles = ["Raw Image","Patch Means","Patch Min Vals","Patch Max Vals","Patch STDs","Patch Skews","Patch Kurtosis"]
days = [4,3,2,1,0]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}. Averaged, then compressed"
    cols_titles = [f"{day} Days Earlier" for day in days]
    rows_titles = [v_title for v_title in values_titles]
    tensors = [events_sequences_avgs[i][day,c,:,:] for day in days] +               [tesnors_patched[v_idx,day,c,:,:] for v_idx in range(6) for day in days] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=5,
                               levels_around_zero=False,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)


events_sequences_compressed_avgs = []
events_sequences_compressed = []
for i,seq_name in enumerate(sequences_names[:-1]):
    sequence = torch.load(sequence_compressed_base_filenames[i]+"inputs.pkl")
    events_sequences_compressed.append(sequence)
    events_sequences_compressed_avgs.append(sequence.mean(0))
    print(f"Loaded sequence #{i} of shape {sequence.shape} -> {events_sequences_compressed_avgs[-1].shape}")


events_sequences[0].shape


i=0
num_events,seq_len,C,H,W = events_sequences[i].shape
reshape_to = [num_events*seq_len,C,H,W]
tesnors_patched_all_events,tensors_values_all_events = calculate_patches_and_values(events_sequences[0].reshape(
                                                                                    reshape_to),
                                                                                    patch_idxs_rows,
                                                                                    patch_idxs_cols,patch_sizes)
tesnors_patched_all_events.shape,tensors_values_all_events.shape


tesnors_patched_all_events_avg = tesnors_patched_all_events.reshape([6,num_events,seq_len,C,H,W]).mean(1)
tesnors_patched_all_events_avg.shape


values_titles = ["Raw Image","Patch Means","Patch Min Vals","Patch Max Vals","Patch STDs","Patch Skews","Patch Kurtosis"]
days = [4,3,2,1,0]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}. Compressed, then averaged"
    cols_titles = [f"{day} Days Earlier" for day in days]
    rows_titles = [v_title for v_title in values_titles]
    tensors = [events_sequences_avgs[i][day,c,:,:] for day in days] +               [tesnors_patched_all_events_avg[v_idx,day,c,:,:] for v_idx in range(6) for day in days] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=5,
                               levels_around_zero=False,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)








sample_idxs = [10]
sample_patched_events = tesnors_patched_all_events.reshape([6,num_events,seq_len,C,H,W])[:,sample_idxs,:,:,:]
sample_patched_events.shape


title_main = f"Events Progression, sequence #{i}: {sequences_names[i]}, Sample Events"

values_titles = ["Patch Means","Patch Min Vals","Patch Max Vals","Patch STDs","Patch Skews","Patch Kurtosis"]
days = [4,3,2,1,0]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}. Sample Event"
    cols_titles = [f"{day} Days Earlier" for day in days]
    rows_titles = [v_title for v_title in values_titles]
    tensors = [sample_patched_events[v_idx,0,day,c,:,:] for v_idx in range(6) for day in days] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=5,
                               levels_around_zero=False,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)

















# Clustering


values_dxs = [0] # Only means

compression_idx = 0
t_compressed = events_sequences_compressed[compression_idx][:,:,values_dxs].flatten(1)
t_compressed.shape


get_kmeans_elbow_scores(t_compressed,max_n_clusters=20,verbose=1,mode="sklearn")








values_dxs = [0] # Only means

compression_idx = 0
t_compressed = events_sequences_compressed[compression_idx][:,:,values_dxs].flatten(1)
t_compressed.shape

num_clusters = 5
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                events_sequences[compression_idx],
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")


t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
len(t_clustered_avgs), t_clustered_avgs[0].shape


title_main = f"Events Progression, sequence #{compression_idx}: {sequences_names[compression_idx]}, {num_clusters} Clusters"

values_titles = ["Patch Means","Patch Min Vals","Patch Max Vals","Patch STDs","Patch Skews","Patch Kurtosis"]
days = [4,3,2,1,0]
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(len(days))] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=5,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=False,
                               cols_titles=cols_titles, rows_titles=rows_titles)














values_dxs = [0,1,2] # Only means,mins,maxs

compression_idx = 0
t_compressed = events_sequences_compressed[compression_idx][:,:,values_dxs].flatten(1)
t_compressed.shape


get_kmeans_elbow_scores(t_compressed,max_n_clusters=20,verbose=0,mode="sklearn")


values_dxs = [0,1,2] # Only means,mins,maxs

compression_idx = 0
t_compressed = events_sequences_compressed[compression_idx][:,:,values_dxs].flatten(1)
t_compressed.shape

num_clusters = 3
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                events_sequences[compression_idx],
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)




title_main = f"Events Progression, sequence #{compression_idx}: {sequences_names[compression_idx]}, {num_clusters} Clusters"
days = [4,3,2,1,0]
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(len(days))] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=5,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=False,
                               cols_titles=cols_titles, rows_titles=rows_titles)














values_dxs = [0,3,4,5] # Only moments

compression_idx = 0
t_compressed = events_sequences_compressed[compression_idx][:,:,values_dxs].flatten(1)
t_compressed.shape


get_kmeans_elbow_scores(t_compressed,max_n_clusters=20,verbose=0,mode="sklearn")


values_dxs = [0,3,4,5] # Only moments

compression_idx = 0
t_compressed = events_sequences_compressed[compression_idx][:,:,values_dxs].flatten(1)
t_compressed.shape


num_clusters = 5
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                events_sequences[compression_idx],
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)




title_main = f"Events Progression, sequence #{compression_idx}: {sequences_names[compression_idx]}, {num_clusters} Clusters"
days = [4,3,2,1,0]
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(len(days))] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=5,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=False,
                               cols_titles=cols_titles, rows_titles=rows_titles)


# Take the wierd group of ~700 events and further cluster them

# Try 5 kmeans clusters, 4 moments, 3X3 patches in the sea, bigger patches elsewhere (sequence 4 light)

















values_dxs = [0] # Only means

compression_idx = 1 # 6 days light
t_compressed = events_sequences_compressed[compression_idx][:,:,values_dxs].flatten(1)
t_compressed.shape


get_kmeans_elbow_scores(t_compressed,max_n_clusters=20,verbose=0,mode="sklearn")


values_dxs = [0] # Only means

compression_idx = 1 # 6 days light
t_compressed = events_sequences_compressed[compression_idx][:,:,values_dxs].flatten(1)
t_compressed.shape



num_clusters = 5
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                events_sequences[compression_idx],
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)




title_main = f"Events Progression, sequence #{compression_idx}: {sequences_names[compression_idx]}, {num_clusters} Clusters"
days = [6,5,4,3,2,1,0]
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(len(days))] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=7,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=False,
                               cols_titles=cols_titles, rows_titles=rows_titles)




