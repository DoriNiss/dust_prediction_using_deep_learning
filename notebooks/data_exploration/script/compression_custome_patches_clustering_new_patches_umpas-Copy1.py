#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/')
from utils.files_loading import *
from utils.data_exploration import *
from utils.meteorology_printing import *
from data_handlers.SequentialHandler import *

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





# debug_year = 2005
# inputs_debug = torch.load(f"{data_dir}/{base_filename}_{debug_year}_input.pkl")
# targets_debug = torch.load(f"{data_dir}/{base_filename}_{debug_year}_target.pkl")
# timestamps_debug = torch.load(f"{data_dir}/{base_filename}_{debug_year}_timestamps.pkl")
# inputs_debug.shape,targets_debug.shape,len(timestamps_debug)


inputs_filename = "dataset_20_81_189_3h_7days_future_all_inputs_pixel_normalized.pkl"
inputs = torch.load(f"{data_dir}/{inputs_filename}")
targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
inputs.shape,targets.shape,len(timestamps)


events_mask = targets[:,0]>=event_threshold
events_idxs = np.arange(events_mask.shape[0])[events_mask]
events_raw = inputs[events_mask]
events_targets = targets[events_mask]
events_timestamps = timestamps[events_idxs]
events_raw.shape,events_targets.shape,len(events_timestamps)


seq_items_idxs = [-4*8,-3*8,-2*8,-1*8,-1*4,0]
seq_handler = SequentialHandler(timestamps,seq_items_idxs,timesteps=[3,"h"],verbose_level=1)


seq_events_inputs,seq_events_targets,seq_events_timestamps =     seq_handler.get_batched_dataset_from_original_idxs(events_idxs,inputs,targets,timestamps)
seq_events_inputs.shape, seq_events_targets.shape, len(seq_events_timestamps)


N,seq_len,C,H,W = seq_events_inputs.shape


seq_events_inputs.shape





# print averages of values


patch_idxs_rows = [
    np.arange(27,51), 
    np.array([17]), 
    np.arange(3,6), 
    np.arange(3,9), 
    np.arange(3,6), 
    np.arange(1,3), 
    np.array([6,7]), 
    np.arange(1,8), 
    np.arange(3), 
    np.arange(1,8), 
]

patch_idxs_cols = [
    np.arange(27*4,27*6), 
    np.arange(9*4,9*6), 
    np.arange(8,12), 
    np.arange(9*5,9*7-6), 
    np.array([18]), 
    np.arange(8,15), 
    np.arange(8,19), 
    np.array([19]), 
    np.arange(2), 
    np.arange(6,8), 
]

patch_sizes =     [
    [1,1],
    [3,3],
    [9,9], 
    [3,3],
    [9,9], 
    [9,9], 
    [9,9], 
    [9,9], 
    [27,27], 
    [9,9], 
]


num_samples = 5
sample_t = inputs[0:5][:,0:1,:,:]
sample_t_patched,sample_t_values = calculate_patches_and_values(sample_t,patch_idxs_rows,patch_idxs_cols,patch_sizes)
print(sample_t_patched.shape, sample_t_values.shape)
v = 0 # mean
for c in range(1):
    tensors = [sample_t_patched[v,i,c] for i in range(num_samples)]+[sample_t[i,c] for i in range(num_samples)]
    print_tensors_with_cartopy(tensors, main_title=f"Sample Patches, {titles_channels_reduced[c]}", titles=None, lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=5,
                               levels_around_zero=False,titles_only_on_edges=False,lock_bar_rows_separately=False,
                               cols_titles=None, rows_titles=None)


1464*2*5


patch_idxs_rows = [
    np.arange(9,17), 
    np.array([17]), 
    np.arange(3,6), 
    np.arange(3,9), 
    np.arange(3,6), 
    np.arange(1,3), 
    np.array([6,7]), 
    np.arange(1,8), 
    np.arange(3), 
    np.arange(1,8), 
]

patch_idxs_cols = [
    np.arange(9*4,9*6), 
    np.arange(9*4,9*6), 
    np.arange(8,12), 
    np.arange(9*5,9*7-6), 
    np.array([18]), 
    np.arange(8,15), 
    np.arange(8,19), 
    np.array([19]), 
    np.arange(2), 
    np.arange(6,8), 
]

patch_sizes =     [
    [3,3],
    [3,3],
    [9,9], 
    [3,3],
    [9,9], 
    [9,9], 
    [9,9], 
    [9,9], 
    [27,27], 
    [9,9], 
]

num_samples = 5
sample_t = inputs[0:5][:,0:1,:,:]
sample_t_patched,sample_t_values = calculate_patches_and_values(sample_t,patch_idxs_rows,patch_idxs_cols,patch_sizes)
print(sample_t_patched.shape, sample_t_values.shape)
v = 0 # mean
for c in range(1):
    tensors = [sample_t_patched[v,i,c] for i in range(num_samples)]+[sample_t[i,c] for i in range(num_samples)]
    print_tensors_with_cartopy(tensors, main_title=f"Sample Patches, {titles_channels_reduced[c]}", titles=None, lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=5,
                               levels_around_zero=False,titles_only_on_edges=False,lock_bar_rows_separately=False,
                               cols_titles=None, rows_titles=None)


312*2*5





seq_events_inputs.shape


channels_for_clustering = [0,16]
print([titles_channels_all_long[c] for c in channels_for_clustering])


seq_events_inputs_for_clustering = seq_events_inputs[:,:,channels_for_clustering].reshape(N*seq_len,len(channels_for_clustering),H,W)
seq_events_inputs_for_clustering.shape


for timing_idx in tqdm([0]): 
    events_patched,events_compressed = calculate_patches_and_values(seq_events_inputs_for_clustering,
                                                                    patch_idxs_rows,patch_idxs_cols,patch_sizes)
print(events_patched.shape,events_compressed.shape)

# events_patched,events_compressed = [],[]
# for seq_idx in tqdm(range(seq_events_inputs.shape[1])):
#     events_patched_item,events_compressed_item = calculate_patches_and_values(seq_events_inputs[:,seq_idx,:,:],
#                                                                               patch_idxs_rows,patch_idxs_cols,
#                                                                               patch_sizes)
#     print(f"Items #{seq_idx} of sequences: {events_patched_item.shape},{events_compressed_item.shape}")
#     events_patched.append(events_patched_item)
#     events_compressed.append(events_compressed_item)





C_clstr = len(channels_for_clustering)


compression_size = events_compressed.shape[2]
events_compressed = events_compressed.reshape(N,seq_len,C_clstr,compression_size,6)
events_compressed.shape


events_patched = events_patched.reshape(6,N,seq_len,C_clstr,H,W)
events_patched.shape


events_patched_avgs = events_patched.mean(1)    


events_patched_avgs.shape


values_titles = ["Means","Mins","Maxs","STDs","Skews","Kurtosis"]
seq_days = [4,3,2,1,0.5,0]
num_seq_days = len(seq_days)

for c_counter,c in enumerate(channels_for_clustering):
    title_main = f"{titles_channels_all_long[c]}"
    tensors = [events_patched_avgs[v,seq_day,c_counter] for v in range(6) for seq_day in range(num_seq_days)]
    titles_rows = values_titles
    titles_cols = [f"{seq_day} Days Earlier" for seq_day in seq_days]
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=num_seq_days,
                               levels_around_zero=False,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=titles_cols, rows_titles=titles_rows)    


seq_events_idxs = seq_handler.get_ramaining_handler_idxs_from_original(events_idxs)
len(seq_events_idxs)











dir_path = f"{data_dir}/compressed/patches_312_seq_6/"

# torch.save(events_patched,dir_path+base_filename+"_events_patched.pkl")
# torch.save(events_compressed,dir_path+base_filename+"_events_compressed.pkl")
# torch.save(events_patched_avgs,dir_path+base_filename+"_events_patched_avgs.pkl")
# torch.save(seq_events_timestamps,dir_path+base_filename+"_seq_events_timestamps.pkl")
# torch.save(seq_events_idxs,dir_path+base_filename+"_seq_events_idxs.pkl")
# torch.save(seq_events_inputs,dir_path+base_filename+"_seq_events_inputs.pkl")
# torch.save(seq_events_targets,dir_path+base_filename+"_seq_events_targets.pkl")


events_patched = torch.load(dir_path+base_filename+"_events_patched.pkl")
events_compressed = torch.load(dir_path+base_filename+"_events_compressed.pkl")
events_patched_avgs = torch.load(dir_path+base_filename+"_events_patched_avgs.pkl")
seq_events_timestamps = torch.load(dir_path+base_filename+"_seq_events_timestamps.pkl")
seq_events_idxs = torch.load(dir_path+base_filename+"_seq_events_idxs.pkl")
seq_events_inputs = torch.load(dir_path+base_filename+"_seq_events_inputs.pkl")
seq_events_targets = torch.load(dir_path+base_filename+"_seq_events_targets.pkl")

print(events_patched.shape,events_compressed.shape,events_patched_avgs.shape)
print(len(seq_events_timestamps),len(seq_events_idxs))
print(seq_events_inputs.shape,seq_events_targets.shape)


events_compressed.shape








# Make patching lighter and faster
# Normalize per value, flatten, umap to N, kmeans
# Prints clusters
# Prints each cluster separately 

# Cluster only by AOD, AOD+SLP, AOD+SLP+PV
# Cluster after: 1-3 channels -> patches -> umap -> kmeans

# Cluster after: SIFT -> PCA : check if gives always the same values, print values next to images -> kmeans


values_idxs = [0] # means

t_compressed = events_compressed[:,:,:,:,values_idxs].flatten(1)
t_compressed.shape


get_kmeans_elbow_scores(t_compressed,max_n_clusters=20,verbose=1,mode="sklearn")


# values_idxs = [0,3,4] # means,STDs,skews
# channels = [0,3,4] # Z,PV,AOD
# patches = [i for i in range(149)]

# t_compressed = events_compressed[:,:,channels][:,:,:,patches][:,:,:,:,values_idxs] #[3253, 7, 3, 149, 3]
# t_compressed_means,t_compressed_stds = t_compressed.mean([0,1,2,3]),t_compressed.std([0,1,2,3])
# t_compressed = (t_compressed-t_compressed_means[None,None,None,None,:])/t_compressed_stds[None,None,None,None,:]
# print(t_compressed.shape)
# t_compressed = t_compressed.flatten(1)

num_clusters = 7
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                seq_events_inputs,
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)


title_main = f"Events Progression, {num_clusters} Clusters"
seq_days = [4,3,2,1,0.5,0]
num_seq_days = len(seq_days)
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(20):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_all_long[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in seq_days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(num_seq_days)] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=num_seq_days,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)











seq_events_targets.shape,len(seq_events_timestamps)





clustered_targets,clustered_timestamps = get_clustered_targets_and_timestamps_from_clusters_idxs(idxs_dict,
                                                                                           seq_events_targets,
                                                                                           seq_events_timestamps)














num_clusters = 4
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                seq_events_inputs,
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)


title_main = f"Events Progression, {num_clusters} Clusters"
seq_days = [4,3,2,1,0.5,0]
num_seq_days = len(seq_days)
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(20):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_all_long[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in seq_days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(num_seq_days)] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=num_seq_days,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)
    
    
clustered_targets,clustered_timestamps = get_clustered_targets_and_timestamps_from_clusters_idxs(idxs_dict,
                                                                                           seq_events_targets,
                                                                                           seq_events_timestamps)








clusters_dir = f"{data_dir}/clusters/4_SLP_AOD_149_patches"

torch.save(x_raw_clustered_dict,clusters_dir+"/events_dict.pkl")
torch.save(clustered_targets,clusters_dir+"/targets_dict.pkl")
torch.save(clustered_timestamps,clusters_dir+"/timestamps_dict.pkl")


clusters_dir




