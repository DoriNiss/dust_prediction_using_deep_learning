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


inputs_filename = "dataset_20_81_189_3h_7days_future_all_inputs_pixel_normalized_reduced_channels.pkl"
inputs = torch.load(f"{data_dir}/{inputs_filename}")
targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
inputs.shape,targets.shape,len(timestamps)


# patch_idxs_rows = [np.array([2]),np.arange(2),np.arange(6)]
# patch_idxs_cols = [np.arange(7), np.arange(3),np.arange(9,21)]
# patch_sizes =     [[27,27],       [27,27],    [9,9]]

patch_idxs_rows = [np.arange(18/3,54/3)]
patch_idxs_cols = [np.arange(99/3,171/3)]
patch_sizes =     [[3,3]]


288*4*3


num_samples = 5
sample_t = inputs[0:5]
sample_t_patched,sample_t_values = calculate_patches_and_values(sample_t,patch_idxs_rows,patch_idxs_cols,patch_sizes)
sample_t_patched.shape, sample_t_values.shape


v = 0 # mean
for c in range(1):
    tensors = [sample_t_patched[v,i,c] for i in range(num_samples)]+[sample_t[i,c] for i in range(num_samples)]
    print_tensors_with_cartopy(tensors, main_title=f"Sample Patches, {titles_channels_reduced[c]}", titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=5,
                               levels_around_zero=False,titles_only_on_edges=False,lock_bar_rows_separately=False,
                               cols_titles=None, rows_titles=None)


# !ls ../../data/datasets_20_81_189_3h_7days_future/metadata


events_mask = targets[:,0]>=event_threshold
events_idxs = np.arange(events_mask.shape[0])[events_mask]
events_raw = inputs[events_mask]
events_targets = targets[events_mask]
events_timestamps = timestamps[events_idxs]
events_raw.shape,events_targets.shape,len(events_timestamps)


seq_items_idxs = [-3*8,-2*8,-1*8,0]
seq_handler = SequentialHandler(timestamps,seq_items_idxs,timesteps=[3,"h"],verbose_level=1)


seq_events_inputs,seq_events_targets,seq_events_timestamps =     seq_handler.get_batched_dataset_from_original_idxs(events_idxs,inputs,targets,timestamps)
seq_events_inputs.shape, seq_events_targets.shape, len(seq_events_timestamps)


N,seq_len,C,H,W = seq_events_inputs.shape


# print averages of values


events_patched,events_compressed = calculate_patches_and_values(seq_events_inputs.reshape(N*seq_len,C,H,W),
                                                                patch_idxs_rows,patch_idxs_cols,patch_sizes)
print(events_patched.shape,events_compressed.shape)


compression_size = events_compressed.shape[2]
events_compressed = events_compressed.reshape(N,seq_len,C,compression_size,6)
events_compressed.shape


events_patched = events_patched.reshape(6,N,seq_len,C,H,W)
events_patched.shape


events_patched_avgs = events_patched.mean(1)    


events_patched_avgs.shape


values_titles = ["Means","Mins","Maxs","STDs","Skews","Kurtosis"]
seq_days = [3,2,1,0]
num_seq_days = len(seq_days)

for c in range(5):
    title_main = f"{titles_channels_reduced[c]}"
    tensors = [events_patched_avgs[v,seq_day,c] for v in range(6) for seq_day in range(num_seq_days)]
    titles_rows = values_titles
    titles_cols = [f"{seq_day} Days Earlier" for seq_day in seq_days]
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=num_seq_days,
                               levels_around_zero=False,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=titles_cols, rows_titles=titles_rows)    


# Cluster from mean,min,max and mean,std,skews
# print only short sequences, than longer sequence
# change patch sizes according to clusters found


seq_events_idxs = seq_handler.get_ramaining_handler_idxs_from_original(events_idxs)
len(seq_events_idxs)











dir_path = f"{data_dir}/compressed/patches_288/"
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


values_idxs = [0,1,2] # means,mins,maxs
channels = [0,3,4] # Z,PV,AOD
patches = [i for i in range(288)]

t_compressed = events_compressed[:,:,channels][:,:,:,patches][:,:,:,:,values_idxs].flatten(1)
t_compressed.shape


get_kmeans_elbow_scores(t_compressed,max_n_clusters=20,verbose=1,mode="sklearn")


num_clusters = 4
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                seq_events_inputs,
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)


title_main = f"Events Progression, {num_clusters} Clusters"
days = [3,2,1,0]
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(len(days))] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=4,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)


values_idxs = [0] # means
channels = [0,3,4] # Z,PV,AOD
patches = [i for i in range(288)]

t_compressed = events_compressed[:,:,channels][:,:,:,patches][:,:,:,:,values_idxs].flatten(1)
t_compressed.shape

num_clusters = 4
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                seq_events_inputs,
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)


title_main = f"Events Progression, {num_clusters} Clusters"
days = [3,2,1,0]
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(len(days))] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=4,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)


values_idxs = [0,3,4] # means,STDs,skews
channels = [0,3,4] # Z,PV,AOD
patches = [i for i in range(288)]

t_compressed = events_compressed[:,:,channels][:,:,:,patches][:,:,:,:,values_idxs].flatten(1)
t_compressed.shape

num_clusters = 4
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                seq_events_inputs,
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)


title_main = f"Events Progression, {num_clusters} Clusters"
days = [3,2,1,0]
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(len(days))] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=4,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)


























###### DIFFERENT PATCHES ######


# patch_idxs_rows = [np.array([0,2]),       np.array([1]),     np.arange(3,6),  np.arange(7,20) ]
# patch_idxs_cols = [np.array([0,1,2,3,6]), np.array([0,1,6]), np.arange(6,12), np.arange(36,54)]
# patch_sizes =     [[27,27],               [27,27],           [9,9],           [3,3]           ]

# patch_idxs_rows = [np.array([0,1,2]), np.arange(1,7) ]
# patch_idxs_cols = [np.array([0,1,6]), np.arange(6,18)]
# patch_sizes =     [[27,27],           [9,9]          ]

# patch_idxs_rows = [np.array([0,1,2]), np.arange(1,7),  np.array([1,5,6]), np.arange(6,15)]
# patch_idxs_cols = [np.array([0,1,6]), np.arange(6,12), np.arange(12,18),  np.arange(36,54)]
# patch_sizes =     [[27,27],           [9,9],           [9,9],             [3,3]]

patch_idxs_rows = [np.arange(6,12), 
                   np.array([1,4]), 
                   np.arange(1,5), 
                   np.array([5]), 
                   np.arange(1,7), 
                   np.arange(3), 
                   np.array([2]), 
                  ]

patch_idxs_cols = [np.arange(42,54), 
                   np.arange(14,18), 
                   np.array([12,13,18,19]), 
                   np.arange(12,20), 
                   np.arange(6,12), 
                   np.arange(2), 
                   np.arange(4,7), 
                  ]

patch_sizes =     [[3,3],   
                   [9,9], 
                   [9,9], 
                   [9,9], 
                   [9,9], 
                   [27,27], 
                   [27,27], 
                  ]

num_samples = 5
sample_t = inputs[10:15]
sample_t_patched,sample_t_values = calculate_patches_and_values(sample_t,patch_idxs_rows,patch_idxs_cols,patch_sizes)
sample_t_patched.shape, sample_t_values.shape


v = 0 # mean
for c in [4]:
    tensors = [sample_t_patched[v,i,c] for i in range(num_samples)]+[sample_t[i,c] for i in range(num_samples)]
    print_tensors_with_cartopy(tensors, main_title=f"Sample Patches, {titles_channels_reduced[c]}", titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=5,
                               levels_around_zero=False,titles_only_on_edges=False,lock_bar_rows_separately=False,
                               cols_titles=None, rows_titles=None)


events_mask = targets[:,0]>=event_threshold
events_idxs = np.arange(events_mask.shape[0])[events_mask]
events_raw = inputs[events_mask]
events_targets = targets[events_mask]
events_timestamps = timestamps[events_idxs]
events_raw.shape,events_targets.shape,len(events_timestamps)


seq_items_idxs = [-6*8,-5*8,-4*8,-3*8,-2*8,-1*8,0]
seq_handler = SequentialHandler(timestamps,seq_items_idxs,timesteps=[3,"h"],verbose_level=1)


seq_events_inputs,seq_events_targets,seq_events_timestamps =     seq_handler.get_batched_dataset_from_original_idxs(events_idxs,inputs,targets,timestamps)
print(seq_events_inputs.shape, seq_events_targets.shape, len(seq_events_timestamps))

N,seq_len,C,H,W = seq_events_inputs.shape


events_patched,events_compressed = calculate_patches_and_values(seq_events_inputs.reshape(N*seq_len,C,H,W),
                                                                patch_idxs_rows,patch_idxs_cols,patch_sizes)
print(events_patched.shape,events_compressed.shape)


compression_size = events_compressed.shape[2]
events_compressed = events_compressed.reshape(N,seq_len,C,compression_size,6)
events_compressed.shape


events_patched = events_patched.reshape(6,N,seq_len,C,H,W)
events_patched.shape


events_patched_avgs = events_patched.mean(1)    
events_patched_avgs.shape


values_titles = ["Means","Mins","Maxs","STDs","Skews","Kurtosis"]
seq_days = [6,5,4,3,2,1,0]
num_seq_days = len(seq_days)

for c in range(5):
    title_main = f"{titles_channels_reduced[c]}"
    tensors = [events_patched_avgs[v,seq_day,c] for v in range(6) for seq_day in range(num_seq_days)]
    titles_rows = values_titles
    titles_cols = [f"{seq_day} Days Earlier" for seq_day in seq_days]
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=num_seq_days,
                               levels_around_zero=False,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=titles_cols, rows_titles=titles_rows)    


seq_events_idxs = seq_handler.get_ramaining_handler_idxs_from_original(events_idxs)
len(seq_events_idxs)


dir_path = f"{data_dir}/compressed/patches_149/"
torch.save(events_patched,dir_path+base_filename+"_events_patched.pkl")
torch.save(events_compressed,dir_path+base_filename+"_events_compressed.pkl")
torch.save(events_patched_avgs,dir_path+base_filename+"_events_patched_avgs.pkl")
torch.save(seq_events_timestamps,dir_path+base_filename+"_seq_events_timestamps.pkl")
torch.save(seq_events_idxs,dir_path+base_filename+"_seq_events_idxs.pkl")
torch.save(seq_events_inputs,dir_path+base_filename+"_seq_events_inputs.pkl")
torch.save(seq_events_targets,dir_path+base_filename+"_seq_events_targets.pkl")


# events_patched = torch.load(dir_path+base_filename+"_events_patched.pkl")
# events_compressed = torch.load(dir_path+base_filename+"_events_compressed.pkl")
# events_patched_avgs = torch.load(dir_path+base_filename+"_events_patched_avgs.pkl")
# seq_events_timestamps = torch.load(dir_path+base_filename+"_seq_events_timestamps.pkl")
# seq_events_idxs = torch.load(dir_path+base_filename+"_seq_events_idxs.pkl")
# seq_events_inputs = torch.load(dir_path+base_filename+"_seq_events_inputs.pkl")
# seq_events_targets = torch.load(dir_path+base_filename+"_seq_events_targets.pkl")

print(events_patched.shape,events_compressed.shape,events_patched_avgs.shape)
print(len(seq_events_timestamps),len(seq_events_idxs))
print(seq_events_inputs.shape,seq_events_targets.shape)


# Normalize all values (std are small...)


values_idxs = [0,3,4] # means,STDs,skews
channels = [0,3,4] # Z,PV,AOD
patches = [i for i in range(149)]

t_compressed = events_compressed[:,:,channels][:,:,:,patches][:,:,:,:,values_idxs] #[3253, 7, 3, 149, 3]
t_compressed_means,t_compressed_stds = t_compressed.mean([0,1,2,3]),t_compressed.std([0,1,2,3])
t_compressed = (t_compressed-t_compressed_means[None,None,None,None,:])/t_compressed_stds[None,None,None,None,:]
print(t_compressed.shape)
t_compressed = t_compressed.flatten(1)

num_clusters = 4
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                seq_events_inputs,
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)


title_main = f"Events Progression, {num_clusters} Clusters"
seq_days = [6,5,4,3,2,1,0]
num_seq_days = len(seq_days)
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in seq_days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(num_seq_days)] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=num_seq_days,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)





values_idxs = [0,1,2,3,4,5] # all + PCA
channels = [0,3,4] # Z,PV,AOD
patches = [i for i in range(149)]

t_compressed = events_compressed[:,:,channels][:,:,:,patches][:,:,:,:,values_idxs] #[3253, 7, 3, 149, 3]
t_compressed_means,t_compressed_stds = t_compressed.mean([0,1,2,3]),t_compressed.std([0,1,2,3])
t_compressed = (t_compressed-t_compressed_means[None,None,None,None,:])/t_compressed_stds[None,None,None,None,:]
print(t_compressed.shape)
t_compressed = t_compressed.flatten(1)
t_compressed = get_pca_compression(t_compressed,N)

num_clusters = 4
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                seq_events_inputs,
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)


title_main = f"Events Progression, {num_clusters} Clusters"
seq_days = [6,5,4,3,2,1,0]
num_seq_days = len(seq_days)
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in seq_days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(num_seq_days)] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=num_seq_days,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)





values_idxs = [0] # means only
channels = [0,3,4] # Z,PV,AOD
patches = [i for i in range(149)]

t_compressed = events_compressed[:,:,channels][:,:,:,patches][:,:,:,:,values_idxs] #[3253, 7, 3, 149, 3]
t_compressed_means,t_compressed_stds = t_compressed.mean([0,1,2,3]),t_compressed.std([0,1,2,3])
t_compressed = (t_compressed-t_compressed_means[None,None,None,None,:])/t_compressed_stds[None,None,None,None,:]
print(t_compressed.shape)
t_compressed = t_compressed.flatten(1)

num_clusters = 5
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                seq_events_inputs,
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)


title_main = f"Events Progression, {num_clusters} Clusters"
seq_days = [6,5,4,3,2,1,0]
num_seq_days = len(seq_days)
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in seq_days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(num_seq_days)] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=num_seq_days,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)


get_kmeans_elbow_scores(t_compressed,max_n_clusters=20,verbose=1,mode="sklearn")


values_idxs = [0,3,4] # means,STDs,skews
channels = [0,3,4] # Z,PV,AOD
patches = [i for i in range(149)]

t_compressed = events_compressed[:,:,channels][:,:,:,patches][:,:,:,:,values_idxs] #[3253, 7, 3, 149, 3]
t_compressed_means,t_compressed_stds = t_compressed.mean([0,1,2,3]),t_compressed.std([0,1,2,3])
t_compressed = (t_compressed-t_compressed_means[None,None,None,None,:])/t_compressed_stds[None,None,None,None,:]
print(t_compressed.shape)
t_compressed = t_compressed.flatten(1)

num_clusters = 3
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                seq_events_inputs,
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)


title_main = f"Events Progression, {num_clusters} Clusters"
seq_days = [6,5,4,3,2,1,0]
num_seq_days = len(seq_days)
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in seq_days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(num_seq_days)] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=num_seq_days,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)


values_idxs = [0,3,4] # means,STDs,skews
channels = [0,3,4] # Z,PV,AOD
patches = [i for i in range(149)]

t_compressed = events_compressed[:,:,channels][:,:,:,patches][:,:,:,:,values_idxs] #[3253, 7, 3, 149, 3]
t_compressed_means,t_compressed_stds = t_compressed.mean([0,1,2,3]),t_compressed.std([0,1,2,3])
t_compressed = (t_compressed-t_compressed_means[None,None,None,None,:])/t_compressed_stds[None,None,None,None,:]
print(t_compressed.shape)
t_compressed = t_compressed.flatten(1)

num_clusters = 5
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                seq_events_inputs,
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)


title_main = f"Events Progression, {num_clusters} Clusters"
seq_days = [6,5,4,3,2,1,0]
num_seq_days = len(seq_days)
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in seq_days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(num_seq_days)] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=num_seq_days,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)


values_idxs = [0,3,4] # means,STDs,skews
channels = [0,3,4] # Z,PV,AOD
patches = [i for i in range(149)]

t_compressed = events_compressed[:,:,channels][:,:,:,patches][:,:,:,:,values_idxs] #[3253, 7, 3, 149, 3]
t_compressed_means,t_compressed_stds = t_compressed.mean([0,1,2,3]),t_compressed.std([0,1,2,3])
t_compressed = (t_compressed-t_compressed_means[None,None,None,None,:])/t_compressed_stds[None,None,None,None,:]
print(t_compressed.shape)
t_compressed = t_compressed.flatten(1)

num_clusters = 6
x_raw_clustered_dict,score,idxs_dict = get_kmeans_clusters_dict(t_compressed,
                                                                seq_events_inputs,
                                                                n_clusters=num_clusters,verbose=1,
                                                                mode="sklearn")
t_clustered_avgs = [x_raw_clustered_dict[n_cluster].mean(0) for n_cluster in x_raw_clustered_dict.keys()]
print(len(t_clustered_avgs), t_clustered_avgs[0].shape)


title_main = f"Events Progression, {num_clusters} Clusters"
seq_days = [6,5,4,3,2,1,0]
num_seq_days = len(seq_days)
clusters = [n_cluster for n_cluster in x_raw_clustered_dict.keys()]

for c in range(5):
    title_main = f"Events Averaged Progression, Patch Compressed, {titles_channels_reduced[c]}, {num_clusters} Clusters"
    cols_titles = [f"{day} Days Earlier" for day in seq_days]
    rows_titles = [f"Cluster {n_cluster}" for n_cluster in clusters]
    tensors = [t_clustered_avgs[n_cluster][day,c,:,:] for n_cluster in clusters for day in range(num_seq_days)] 
    print_tensors_with_cartopy(tensors, main_title=title_main, titles=None, lock_bar=False, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,num_cols=num_seq_days,
                               levels_around_zero=True,titles_only_on_edges=True,lock_bar_rows_separately=True,
                               cols_titles=cols_titles, rows_titles=rows_titles)


get_kmeans_elbow_scores(t_compressed,max_n_clusters=20,verbose=1,mode="sklearn")







