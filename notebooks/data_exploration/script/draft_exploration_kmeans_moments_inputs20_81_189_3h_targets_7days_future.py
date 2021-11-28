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
description = torch.load(f"{data_dir}/metadata/dataset_20_81_189_3h_7days_future_2003_descriptions.pkl")[0]
titles_channels = [description["input"][i]["long"] for i in range(20)]
event_threshold = 73.4


inputs = torch.load(f"{data_dir}/{base_filename}_all_inputs.pkl")
targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
inputs.shape,targets.shape,len(timestamps)














all_channels_averages = inputs.mean([0])
inputs_minus_avgs = inputs-all_channels_averages[None,:,:,:]
all_channels_averages.shape,inputs_minus_avgs.shape


t_events_only = inputs_minus_avgs[targets[:,0]>=event_threshold]
t_events_only.shape


t = t_events_only
for c in range(t.shape[1]):
    print(c,get_min_num_of_sift_features(t[:,c,:,:],contrastThreshold=0.005,edgeThreshold=20))


print([f"{i}:{description['input'][i]['short']}" for i in range(20)])


channels_to_average = [np.arange(1),
                       np.array([4,5,6,18]),np.array([7,8,9,19]),
                       np.arange(10,14),np.array([14])]

print(len(channels_to_average),channels_to_average)
for c in range(len(channels_to_average)):
    print(f"{c}:\n   {[description['input'][i]['short'] for i in channels_to_average[c]]}")


t_averaged_channels_events_only = average_related_channels(t_events_only,channels_to_average)


t = t_averaged_channels_events_only
for c in range(t.shape[1]):
    print(c,get_min_num_of_sift_features(t[:,c,:,:],contrastThreshold=0.04,edgeThreshold=10))


for c in range(t.shape[1]):
    print(c,get_min_num_of_sift_features(t[:,c,:,:],contrastThreshold=0.005,edgeThreshold=20))


t_samples = t_averaged_channels_events_only[:10,:,:,:]
t_samples.shape


t_samples_compression = get_sift_pca_compressed_simple(t_samples,n_descriptors=30,
                                                       n_components_xy=5, n_components_descriptors=5,
                                   to_normalize=True,contrastThreshold=0.005,edgeThreshold=20)
t_samples_compression.shape


t_samples_compression = get_sift_pca_compressed_simple(t_samples,n_descriptors=0,
                                                       n_components_xy=0, n_components_descriptors=0,
                                   to_normalize=True,contrastThreshold=0.04,edgeThreshold=10)
t_samples_compression.shape











# compressed_moments_0: 27*27, weighted False, add_max_patch_idxs False, add_min_max_positions False
# compressed_moments_1: 27*27, weighted True, add_max_patch_idxs False, add_min_max_positions False
# compressed_moments_2: 27*27, weighted False, add_max_patch_idxs False, add_min_max_positions True
# compressed_moments_3: 27*27, weighted True, add_max_patch_idxs True, add_min_max_positions True
# compressed_moments_4: sift moments - no wieghts
# compressed_moments_5: sift moments - mean-weighted
# compressed_moments_6: sift - concatenated xy and descriptors of all channels 
# compressed_moments_7: sift - concatenated xy only of all channels 
# compressed_moments_8: sift,pca compressed 30*2 -> 25, 30*128 -> 25 (N,40)

# t = t_averaged_channels_events_only
t_averaged_channels = average_related_channels(inputs_minus_avgs,channels_to_average)
t = t_averaged_channels

patch_shape = (27,27)
print("Calculating compressed_moments_0...")
compressed_moments_0 = get_compressed_patch_moments_average(t, 
                                                            patch_shape,
                                                            weight_with_maxes=False, 
                                                            add_max_patch_idxs=False,
                                                            add_min_max_positions=False)
print(f"..Done! Shape: {compressed_moments_0.shape}")

print("Calculating compressed_moments_1...")
compressed_moments_1 = get_compressed_patch_moments_average(t, 
                                                            patch_shape,
                                                            weight_with_maxes=True, 
                                                            add_max_patch_idxs=False,
                                                            add_min_max_positions=False)
print(f"..Done! Shape: {compressed_moments_1.shape}")

print("Calculating compressed_moments_2...")
compressed_moments_2 = get_compressed_patch_moments_average(t, 
                                                            patch_shape,
                                                            weight_with_maxes=False, 
                                                            add_max_patch_idxs=False,
                                                            add_min_max_positions=True)
print(f"..Done! Shape: {compressed_moments_2.shape}")

print("Calculating compressed_moments_3...")
compressed_moments_3 = get_compressed_patch_moments_average(t,
                                                            patch_shape,
                                                            weight_with_maxes=True, 
                                                            add_max_patch_idxs=True,
                                                            add_min_max_positions=True)
print(f"..Done! Shape: {compressed_moments_3.shape}")

print("Calculating compressed_moments_4...")
compressed_moments_4 = get_compressed_sift_moments(t,
                                                   n_features=0,contrastThreshold=0.005,edgeThreshold=20,
                                                   weight_descriptors_by_descriptors_means=False,
                                                   weight_keypoints_by_descriptors_means=False)
print(f"..Done! Shape: {compressed_moments_4.shape}")

print("Calculating compressed_moments_5...")
compressed_moments_5 = get_compressed_sift_moments(t,
                                                   n_features=0,contrastThreshold=0.005,edgeThreshold=20,
                                                   weight_descriptors_by_descriptors_means=True,
                                                   weight_keypoints_by_descriptors_means=True)
print(f"..Done! Shape: {compressed_moments_5.shape}")

# print("Calculating compressed_moments_6...")
# compressed_moments_6 = get_sift_pca_compressed_simple(t,n_descriptors=30,
#                                                       n_components_xy=0, n_components_descriptors=0,
#                                                       to_normalize=True,contrastThreshold=0.04,edgeThreshold=10)
# print(f"..Done! Shape: {compressed_moments_6.shape}")

print("Calculating compressed_moments_7...")
sift_xy,_ = get_all_sifts_per_channel(t,verbose=1,contrastThreshold=0.005,edgeThreshold=20) #C*[N,min_num_features,2]
compressed_moments_7 = torch.cat([sift_xy[c].flatten(1).float() for c in range(t.shape[1])],dim=1)
print(f"..Done! Shape: {compressed_moments_7.shape}")

# print("Calculating compressed_moments_8...")
# compressed_moments_8 = get_sift_pca_compressed_simple(t,n_descriptors=30,
#                                                       n_components_xy=25, n_components_descriptors=25,
#                                                       to_normalize=True,contrastThreshold=0.005,edgeThreshold=20)
# print(f"..Done! Shape: {compressed_moments_8.shape}")


print("Calculating compressed_moments_8...")
compressed_moments_8 = get_sift_pca_compressed_simple(t,n_descriptors=20,
                                                      n_components_xy=25, n_components_descriptors=25,
                                                      to_normalize=True,contrastThreshold=0.005,edgeThreshold=20)
print(f"..Done! Shape: {compressed_moments_8.shape}")














# torch.save(compressed_moments_0,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_0.pkl")
# torch.save(compressed_moments_1,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_1.pkl")
# torch.save(compressed_moments_2,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_2.pkl")
# torch.save(compressed_moments_3,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_3.pkl")
# torch.save(compressed_moments_4,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_4.pkl")
# torch.save(compressed_moments_5,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_5.pkl")
# # torch.save(compressed_moments_6,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_6.pkl")
# torch.save(compressed_moments_7,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_7.pkl")
# torch.save(compressed_moments_8,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_8.pkl")





# print("Calculating compressed_moments_8...")
# sift_xy,sift_descriptors = get_all_sifts_per_channel(inputs_minus_avgs_choosen_channels,verbose=1) 
# #C*[N,min_num_features,2],C*[N,min_num_features,128]
# t_xy = torch.cat([sift_xy[c].flatten(1)
#                   for c in range(inputs_minus_avgs_choosen_channels.shape[1])],dim=1)
# t_desc = torch.cat([sift_descriptors[c].flatten(1)
#                     for c in range(inputs_minus_avgs_choosen_channels.shape[1])],dim=1)
# compressed_moments_8 = torch.cat([t_xy,t_desc],dim=1)
# print(f"..Done! Shape: {compressed_moments_8.shape}")

# torch.save(compressed_moments_8,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_8.pkl")


compressed_moments_0 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_0.pkl")
compressed_moments_1 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_1.pkl")
compressed_moments_2 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_2.pkl")
compressed_moments_3 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_3.pkl")
compressed_moments_4 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_4.pkl")
compressed_moments_5 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_5.pkl")
# compressed_moments_6 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_6.pkl")
compressed_moments_7 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_7.pkl")
compressed_moments_8 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_8.pkl")
compressed_moments_0.shape,compressed_moments_1.shape,compressed_moments_2.shape,compressed_moments_3.shape, compressed_moments_4.shape,compressed_moments_5.shape,compressed_moments_7.shape, compressed_moments_8.shape


compressed_moments_7 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_7.pkl")
compressed_moments_8 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_8.pkl")
compressed_moments_7.shape,compressed_moments_8.shape











seq_items_idxs = [-5*8,-4*8,-3*8,-2*8,-1*8,0]
seq_handler = SequentialHandler(timestamps,seq_items_idxs,timesteps=[3,"h"],verbose_level=1)


len(timestamps),inputs.shape,targets.shape#,t.shape


timestamps[seq_handler.rows_that_have_sequences],len(timestamps[seq_handler.rows_that_have_sequences])


events_idxs_original = targets[:,0]>=event_threshold
# clear_idxs_original = targets[:,0]<event_threshold
events_idxs_original_list = np.arange(len(events_idxs_original))[events_idxs_original]
# clear_idxs_original_list = np.arange(len(clear_idxs_original))[clear_idxs_original]
# print(events_idxs_original.shape,clear_idxs_original.shape,events_idxs_original.sum(),clear_idxs_original.sum(), 
#       len(events_idxs_original_list),len(clear_idxs_original_list))
print(events_idxs_original.shape,events_idxs_original.sum(),len(events_idxs_original_list))


events_handler_idxs_that_have_sequences = [seq_handler.translate_original_idx_to_handler(i) 
                                           for i in tqdm(events_idxs_original_list)] 
events_handler_idxs_that_have_sequences = [i for i in events_handler_idxs_that_have_sequences if i is not None]
len(events_handler_idxs_that_have_sequences) 


idx = events_handler_idxs_that_have_sequences[-1]
seq_handler.translate_handler_idx_to_original(idx), timestamps[seq_handler.translate_handler_idx_to_original(idx)]





# Need to compress all inputs to get sequences... TODO: function that creates a sequence and compresses

compressed_moments_sequences = []
for i,compressed_moments in enumerate([
#                                        compressed_moments_0,compressed_moments_1,
#                                        compressed_moments_2,compressed_moments_3,
#                                        compressed_moments_4,compressed_moments_5,
#                                        compressed_moments_6,
                                       compressed_moments_7,
                                       compressed_moments_8,
                                      ]):
    compressed_moments_events_only=seq_handler.get_batched_sequences_from_original_idxs(compressed_moments,
                                                                                        events_idxs_original_list,
                                                                                        add_dim=True) 
    compressed_moments_sequences.append(compressed_moments_events_only)
    print(i,compressed_moments_events_only.shape,compressed_moments_sequences[i].shape)











# idx = 40
# for i in range(4):
#     print(i)
#     for s in range(4):
#         print(seq_items_idxs[s])
#         plt.imshow(compressed_moments_sequences[i][idx,s])
#         plt.show()








# std scores, fast_pytorch clustering


for i,t in enumerate(compressed_moments_sequences):
    n = i if i<6 else i+1
    print(f"\n\n#### Compression {n}:")
    if i<6:
        t_normed,_,_ = normalize_channels_averages(t.transpose(1,2))
    else:
        t_normed = (t-t.mean())/t.std()
    t_flat = t_normed.flatten(1)
    get_kmeans_elbow_scores(t_flat,max_n_clusters=20,verbose=1,mode="fast_pytorch")








# sklearn score method, full batch


for i,t in enumerate(compressed_moments_sequences):
#     n = i if i<6 else i+1
#     print(f"\n\n#### Compression {n}:")
#     if i<6:
#         t_normed,_,_ = normalize_channels_averages(t.transpose(1,2))
#     else:
    t_normed = (t-t.mean())/t.std()
    t_flat = t_normed.flatten(1)
    get_kmeans_elbow_scores(t_flat,max_n_clusters=20,verbose=1,mode="sklearn")








# sklearn score method, minibatch


for i,t in enumerate(compressed_moments_sequences):
    n = i if i<6 else i+1
    print(f"\n\n#### Compression {n}:")
    if i<6:
        t_normed,_,_ = normalize_channels_averages(t.transpose(1,2))
    else:
        t_normed = (t-t.mean())/t.std()
    t_flat = t_normed.flatten(1)
    get_kmeans_elbow_scores(t_flat,max_n_clusters=20,verbose=1,mode="sklearn_minibatch")

















all_channels_averages = inputs.mean([0])
inputs_minus_avgs = inputs-all_channels_averages[None,:,:,:]
all_channels_averages.shape,inputs_minus_avgs.shape


inputs_sequences_events_only=seq_handler.get_batched_sequences_from_original_idxs(inputs_minus_avgs,
                                                                                  events_idxs_original_list,
                                                                                  add_dim=True) 
inputs_sequences_events_only.shape





# torch.save(inputs_sequences_events_only,
#            f"{data_dir_compressed}/{base_filename}_inputs_20channels_sequences_6days_events_only.pkl")











[seq_items_idxs[i]/8 for i in range(6)]


num_clusters = 5


# i = -1
# cluster_compression = compressed_moments_sequences[i]
# if i<6 and i>-1:
#     t_normed,_,_ = normalize_channels_averages(t.transpose(1,2))
# else:

t = compressed_moments_sequences[0]

t_normed = (t-t.mean())/t.std()
t_flat = t_normed.flatten(1)
clusters,_,idxs_dict = get_kmeans_clusters_dict(t_flat,
                                                inputs_sequences_events_only,
                                                num_clusters,
                                                verbose=1,
                                                mode="sklearn")


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
    print(f"{i}:{clusters_sequence_avg[i].shape}")


# channels_titles = [description["input"][i]["short"] for i in range(20)]
# titles = [f"{-i} Days Earlier, C{cluster}" for cluster in range(num_clusters) for i in range(-5,1)]
# title_base = f"Average Normalized Events Anonamlies Progression, {num_clusters} Clusters"


# for c in range(20):
#     levels_around_zero = c<10 or c>=18
#     title = f"{title_base}: {titles_channels[c]}"
#     print(title)
#     tensors = [clusters_sequence_avg[cluster][seq_i,c] for cluster in range(num_clusters) for seq_i in range(6)]
#     print_tensors_with_cartopy(tensors, main_title=title, titles=titles, lock_bar=True, 
#                                lons=None, lats=None, num_levels=10, manual_levels=None,
#                                levels_around_zero=levels_around_zero, num_cols=6)  














num_clusters = 5
# i = 6
# cluster_compression = compressed_moments_sequences[i]
# if i<6 and i>-1:
#     t_normed,_,_ = normalize_channels_averages(t.transpose(1,2))
# else:
#     t_normed = (t-t.mean())/t.std()

t = compressed_moments_sequences[1]

t_normed = (t-t.mean())/t.std()

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
    print(f"{i}:{clusters_sequence_avg[i].shape}, averaged from tensors of shape {clusters[i].shape}")
    
# Normalization:
for i in range(num_clusters):
    means = clusters_sequence_avg[i].mean([0,2,3])
    stds = clusters_sequence_avg[i].std([0,2,3])
    clusters_sequence_avg[i] = (clusters_sequence_avg[i]-means[None,:,None,None])/stds[None,:,None,None]
    print(f"{i}:,{clusters_sequence_avg[i].shape}")
    
channels_titles = [description["input"][i]["short"] for i in range(20)]
titles = [f"{-i} Days Earlier, C{cluster}" for cluster in range(num_clusters) for i in range(-5,1)]
title_base = f"Average Normalized Events Anonamlies Progression, {num_clusters} Clusters"


for c in range(20):
    levels_around_zero = c<10 or c>=18
    title = f"{title_base}: {titles_channels[c]}"
    print(title)
    tensors = [clusters_sequence_avg[cluster][seq_i,c] for cluster in range(num_clusters) for seq_i in range(6)]
    print_tensors_with_cartopy(tensors, main_title=title, titles=titles, lock_bar=True, 
                               lons=None, lats=None, num_levels=10, manual_levels=None,
                               levels_around_zero=levels_around_zero, num_cols=6)  











idxs_events_handler = [seq_handler.translate_original_idx_to_handler(idx)
                        for idx in events_idxs_original_list]
idxs_events_handler = [idx for idx in idxs_events_handler if idx is not None]
idxs_events_handler
idxs_events_original_with_sequences = [seq_handler.translate_handler_idx_to_original(idx)
                                       for idx in idxs_events_handler]
len(idxs_events_original_with_sequences), idxs_events_original_with_sequences[-1]


timestamps_events_with_sequences = timestamps[idxs_events_original_with_sequences]
timestamps_events_with_sequences


len(timestamps_events_with_sequences)











timestamps_clusters_dict = {cluster_label: timestamps_events_with_sequences[idxs_dict[cluster_label]]
                            for cluster_label in range(num_clusters)}

clusters_months_counters = {cluster_label:{i+1:0 for i in range(13)} for cluster_label in range(num_clusters)}
clusters_months_counters
for cluster_label in range(num_clusters):
    timestamps_cluster = timestamps_clusters_dict[cluster_label]
    for t in timestamps_cluster:
        clusters_months_counters[cluster_label][t.month]+=1

clusters_num_events_plots = {}
for cluster_label in range(num_clusters):
    num_events = []
    for m in range(1,13):
        num_events.append(clusters_months_counters[cluster_label][m])
    clusters_num_events_plots[cluster_label] = np.array(num_events)
    
plt.figure()
for cluster_label in range(num_clusters):
    plt.plot(np.arange(1,13),clusters_num_events_plots[cluster_label],label=f"{cluster_label}")
plt.legend(bbox_to_anchor=(1.2,1), loc="upper right")
plt.show()














# n_cols = 6
# n_rows = 2
# tensors_test = [tensors[i] for i in range(n_cols)]*n_rows
# titles_test = [titles[i] for i in range(n_cols)]*n_rows
# print_tensors_with_cartopy(tensors_test, main_title=title, titles=titles_test, lock_bar=True, 
#                            lons=None, lats=None, num_levels=10, manual_levels=None,
#                            levels_around_zero=levels_around_zero, num_rows=n_rows, num_cols=n_cols)  







