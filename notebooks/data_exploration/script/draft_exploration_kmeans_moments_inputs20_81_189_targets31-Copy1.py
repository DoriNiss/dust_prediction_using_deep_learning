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
data_dir_compressed = data_dir+"/compressed"
base_filename = "dataset_20_81_189_3h_7days_future"
description = torch.load(f"{data_dir}/metadata/dataset_20_81_189_3h_7days_future_2003_descriptions.pkl")
titles_channels = [description["input"][i]["long"] for i in range(20)]
event_threshold = 73.4





inputs = torch.load(f"{data_dir}/{base_filename}_all_inputs.pkl")
targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
targets.shape,len(timestamps)





# description["input"]


choosen_channels = np.array([0,3,4,7,13,14])
choosen_channels_titles = [description["input"][i]["short"] for i in choosen_channels]
choosen_channels_titles


# all_channels_averages = inputs.mean([0])
# inputs_minus_avgs = inputs-all_channels_averages[None,:,:,:]
# all_channels_averages.shape,inputs_minus_avgs.shape


# inputs_minus_avgs_choosen_channels = inputs_minus_avgs[:,choosen_channels,:,:]
# inputs_minus_avgs_choosen_channels.shape


# torch.save(inputs_minus_avgs_choosen_channels,f"{data_dir_compressed}/{base_filename}_inputs_minus_avgs_6_channels.pkl")
inputs_minus_avgs_choosen_channels = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_minus_avgs_6_channels.pkl")
inputs_minus_avgs_choosen_channels.shape





# # compressed_moments_0: 27*27, weighted False, add_max_patch_idxs False, add_min_max_positions False
# # compressed_moments_1: 27*27, weighted True, add_max_patch_idxs False, add_min_max_positions False
# # compressed_moments_2: 27*27, weighted False, add_max_patch_idxs False, add_min_max_positions True
# # compressed_moments_3: 27*27, weighted True, add_max_patch_idxs True, add_min_max_positions True

# patch_shape = (27,27)
# print("Calculating compressed_moments_0...")
# compressed_moments_0 = get_compressed_patch_moments_average(inputs_minus_avgs_choosen_channels, 
#                                                             patch_shape,
#                                                             weight_with_maxes=False, 
#                                                             add_max_patch_idxs=False,
#                                                             add_min_max_positions=False)
# print(f"..Done! Shape: {compressed_moments_0.shape}")
# print("Calculating compressed_moments_1...")
# compressed_moments_1 = get_compressed_patch_moments_average(inputs_minus_avgs_choosen_channels, 
#                                                             patch_shape,
#                                                             weight_with_maxes=True, 
#                                                             add_max_patch_idxs=False,
#                                                             add_min_max_positions=False)
# print(f"..Done! Shape: {compressed_moments_1.shape}")
# print("Calculating compressed_moments_2...")
# compressed_moments_2 = get_compressed_patch_moments_average(inputs_minus_avgs_choosen_channels, 
#                                                             patch_shape,
#                                                             weight_with_maxes=False, 
#                                                             add_max_patch_idxs=False,
#                                                             add_min_max_positions=True)
# print(f"..Done! Shape: {compressed_moments_2.shape}")
# print("Calculating compressed_moments_3...")
# compressed_moments_3 = get_compressed_patch_moments_average(inputs_minus_avgs_choosen_channels, 
#                                                             patch_shape,
#                                                             weight_with_maxes=True, 
#                                                             add_max_patch_idxs=True,
#                                                             add_min_max_positions=True)
# print(f"..Done! Shape: {compressed_moments_3.shape}")


# torch.save(compressed_moments_0,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_0.pkl")
# torch.save(compressed_moments_1,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_1.pkl")
# torch.save(compressed_moments_2,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_2.pkl")
# torch.save(compressed_moments_3,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_3.pkl")


compressed_moments_0 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_0.pkl")
compressed_moments_1 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_1.pkl")
compressed_moments_2 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_2.pkl")
compressed_moments_3 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_3.pkl")
compressed_moments_0.shape,compressed_moments_1.shape,compressed_moments_2.shape,compressed_moments_3.shape


seq_items_idxs = [-6*4,-4*4,-2*4,0]
seq_handler = SequentialHandler(timestamps,seq_items_idxs,timesteps=[6,"h"],verbose_level=1)


events_idxs_original = targets[:,-1]==1
clear_idxs_original = targets[:,-1]==0
events_idxs_original_list = np.arange(len(events_idxs_original))[events_idxs_original]
clear_idxs_original_list = np.arange(len(clear_idxs_original))[clear_idxs_original]
print(events_idxs_original.shape,clear_idxs_original.shape,events_idxs_original.sum(),clear_idxs_original.sum(), 
      len(events_idxs_original_list),len(clear_idxs_original_list))


events_handler_idxs_that_have_sequences = [seq_handler.translate_original_idx_to_handler(i) 
                                           for i in tqdm(events_idxs_original_list)] 
events_handler_idxs_that_have_sequences = [i for i in events_handler_idxs_that_have_sequences if i is not None]
len(events_handler_idxs_that_have_sequences) 


compressed_moments_sequences = []
for i,compressed_moments in enumerate([compressed_moments_0,compressed_moments_1,
                                       compressed_moments_2,compressed_moments_3]):
    compressed_moments_events_only=seq_handler.get_batched_sequences_from_original_idxs(compressed_moments,
                                                                                        events_idxs_original_list,
                                                                                        add_dim=True) 
    compressed_moments_sequences.append(compressed_moments_events_only)
    print(i,compressed_moments_sequences[i].shape)











# idx = 40
# for i in range(4):
#     print(i)
#     for s in range(4):
#         print(seq_items_idxs[s])
#         plt.imshow(compressed_moments_sequences[i][idx,s])
#         plt.show()








# std scores, fast_pytorch clustering


for i,t in enumerate(compressed_moments_sequences):
    print(f"\n\n#### Compression {i}:")
    t_normed,_,_ = normalize_channels_averages(t.transpose(1,2))
    t_flat = t_normed.flatten(1)
    get_kmeans_elbow_scores(t_flat,max_n_clusters=20,verbose=1,mode="fast_pytorch")








# sklearn score method, full batch


for i,t in enumerate(compressed_moments_sequences):
    print(f"\n\n#### Compression {i}:")
    t_normed,_,_ = normalize_channels_averages(t.transpose(1,2))
    t_flat = t_normed.flatten(1)
    get_kmeans_elbow_scores(t_flat,max_n_clusters=20,verbose=1,mode="sklearn")








# sklearn score method, minibatch


for i,t in enumerate(compressed_moments_sequences):
    print(f"\n\n#### Compression {i}:")
    t_normed,_,_ = normalize_channels_averages(t.transpose(1,2))
    t_flat = t_normed.flatten(1)
    get_kmeans_elbow_scores(t_flat,max_n_clusters=20,verbose=1,mode="sklearn_minibatch")








all_channels_averages = inputs.mean([0])
inputs_minus_avgs = inputs-all_channels_averages[None,:,:,:]
all_channels_averages.shape,inputs_minus_avgs.shape


inputs_sequences_events_only=seq_handler.get_batched_sequences_from_original_idxs(inputs_minus_avgs,
                                                                                  events_idxs_original_list,
                                                                                  add_dim=True) 
inputs_sequences_events_only.shape








num_clusters = 4


cluster_compression = compressed_moments_sequences[1]
t_normed,_,_ = normalize_channels_averages(cluster_compression.transpose(1,2))
t_flat = t_normed.flatten(1)
clusters,_,idxs_dict = get_kmeans_clusters_dict(t_flat,
                                                inputs_sequences_events_only,
                                                num_clusters,
                                                verbose=1,
                                                mode="fast_pytorch")


print("Resulting shapes:")
for i in range(num_clusters):
    print(f"{i}:{clusters[i].shape}")


clusters_sequence_avg = {}
for i in range(num_clusters):
    clusters_sequence_avg[i] = clusters[i].mean(0)
    print(f"{i}:,{clusters_sequence_avg[i].shape}, averaged from tensors of shape {clusters[i].shape}")


# Normalization:
for i in range(num_clusters):
    means = clusters_sequence_avg[i].mean([0,2,3])
    stds = clusters_sequence_avg[i].std([0,2,3])
    clusters_sequence_avg[i] = (clusters_sequence_avg[i]-means[None,:,None,None])/stds[None,:,None,None]
    print(f"{i}:,{clusters_sequence_avg[i].shape}")


channels_titles = [description["input"][i]["short"] for i in range(20)]
titles = [f"{-i} Days Before Event, Cluster {cluster}" for cluster in range(num_clusters) for i in range(-6,1,2)]
title_base = f"Average Normalized Events Anonamlies Progression, {num_clusters} Clusters"


for c in range(20):
    levels_around_zero = c<10 or c>=18
    title = f"{title_base}: {titles_channels[c]}"
    print(title)
    tensors = [clusters_sequence_avg[cluster][seq_i,c] for cluster in range(num_clusters) for seq_i in range(4)]
    print_tensors_with_cartopy(tensors, main_title=title, titles=titles, lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,
                               levels_around_zero=levels_around_zero)  







