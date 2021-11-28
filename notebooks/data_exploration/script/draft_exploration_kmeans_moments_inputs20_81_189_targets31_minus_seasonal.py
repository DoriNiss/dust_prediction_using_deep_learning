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
data_dir_compressed = data_dir+"/compressed"
base_filename = "dataset_20_81_189_averaged_dust_24h"
description = torch.load(f"{data_dir}/metadata/dataset_20_81_189_averaged_dust_24h_metadata.pkl")
titles_channels = [description["input"][i]["short"] for i in range(20)]
inputs_minus_seasonal_path = f"{data_dir}/{base_filename}_inputs_minus_seasonal.pkl"
inputs_minus_seasonal_path = f"{data_dir}/{base_filename}_inputs_minus_seasonal.pkl"
event_threshold = 73.4


# inputs_minus_seasonal = torch.load(inputs_minus_seasonal_path)
# inputs_minus_seasonal.shape


targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
targets.shape,len(timestamps)


# description["input"]


choosen_channels = np.array([0,3,4,7,13,14])
choosen_channels_titles = [description["input"][i]["short"] for i in choosen_channels]
choosen_channels_titles


# inputs_minus_seasonal_choosen_channels = inputs_minus_seasonal[:,choosen_channels,:,:]
# inputs_minus_seasonal_choosen_channels.shape


# torch.save(inputs_minus_seasonal_choosen_channels,f"{data_dir_compressed}/{base_filename}_inputs_minus_seasonal_6_channels.pkl")
inputs_minus_seasonal_choosen_channels = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_minus_seasonal_6_channels.pkl")
inputs_minus_seasonal_choosen_channels.shape





# compressed_moments_0: 27*27, weighted False, add_max_patch_idxs False, add_min_max_positions False
# compressed_moments_1: 27*27, weighted True, add_max_patch_idxs False, add_min_max_positions False
# compressed_moments_2: 27*27, weighted False, add_max_patch_idxs False, add_min_max_positions True
# compressed_moments_3: 27*27, weighted True, add_max_patch_idxs True, add_min_max_positions True

patch_shape = (27,27)
print("Calculating compressed_moments_0")
compressed_moments_0 = get_compressed_patch_moments_average(inputs_minus_seasonal_choosen_channels, 
                                                            patch_shape,
                                                            weight_with_maxes=False, 
                                                            add_max_patch_idxs=False,
                                                            add_min_max_positions=False)
print("Calculating compressed_moments_1")
compressed_moments_1 = get_compressed_patch_moments_average(inputs_minus_seasonal_choosen_channels, 
                                                            patch_shape,
                                                            weight_with_maxes=True, 
                                                            add_max_patch_idxs=False,
                                                            add_min_max_positions=False)
print("Calculating compressed_moments_2")
compressed_moments_2 = get_compressed_patch_moments_average(inputs_minus_seasonal_choosen_channels, 
                                                            patch_shape,
                                                            weight_with_maxes=False, 
                                                            add_max_patch_idxs=False,
                                                            add_min_max_positions=True)
print("Calculating compressed_moments_3")
compressed_moments_3 = get_compressed_patch_moments_average(inputs_minus_seasonal_choosen_channels, 
                                                            patch_shape,
                                                            weight_with_maxes=True, 
                                                            add_max_patch_idxs=True,
                                                            add_min_max_positions=True)


compressed_moments_0.shape,compressed_moments_1.shape,compressed_moments_2.shape,compressed_moments_3.shape


torch.save(compressed_moments_0,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_0.pkl")
torch.save(compressed_moments_1,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_1.pkl")
torch.save(compressed_moments_2,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_2.pkl")
torch.save(compressed_moments_3,f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_3.pkl")


# compressed_moments_0 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_0.pkl")
# compressed_moments_1 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_1.pkl")
# compressed_moments_2 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_2.pkl")
# compressed_moments_3 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_3.pkl")











from data_handlers.SequentialHandler import *


seq_items_idxs = [-6*4,-4*4,-2*4,0]


seq_handler = SequentialHandler(timestamps,seq_items_idxs,timesteps=[6,"h"],verbose_level=1)


timestamps[seq_handler.rows_that_have_sequences[0]]


timestamps_that_have_sequenc = timestamps[seq_handler.rows_that_have_sequences]
len(timestamps_that_have_sequenc),timestamps_that_have_sequenc[0]


events_idxs = targets[:,-1]==1
clear_idxs = targets[:,-1]==0
print(events_idxs.shape,clear_idxs.shape,events_idxs.sum(),clear_idxs.sum())


events_idxs_list = []
clear_idxs_list = []
for i,is_event in enumerate(events_idxs):
    if is_event:
        events_idxs_list.append(i)
    else:
        clear_idxs_list.append(i)
len(events_idxs_list),len(clear_idxs_list)


len(timestamps[events_idxs_list]),timestamps[events_idxs_list][0]


events_idxs_that_have_sequences = []
for event_idx in events_idxs_list:
    if timestamps[event_idx] in timestamps_that_have_sequenc:
        events_idxs_that_have_sequences.append(event_idx)
print(len(events_idxs_that_have_sequences),events_idxs_that_have_sequences[0])


compressed_moments_0[events_idxs_that_have_sequences].shape


events_idxs_that_have_sequences[-1]


targets_sequences = seq_handler.get_all_sequences(targets,add_dim=True)
targets_sequences.shape


compressed_moments_sequences = []
for i,compressed_moments in enumerate([compressed_moments_0,compressed_moments_1,
                                       compressed_moments_2,compressed_moments_3]):
    compressed_moments_all = seq_handler.get_all_sequences(compressed_moments,add_dim=True)
    compressed_moments_events_only = compressed_moments_all[targets_sequences[:,-1,-1]==1]
    compressed_moments_sequences.append(compressed_moments_events_only)

print([compressed_moments_sequences[i].shape for i in range(4)])








# compressed_moments_sequences = [compressed_moments_0_sequences,compressed_moments_1_sequences,
#                                 compressed_moments_2_sequences,compressed_moments_3_sequences]


# idx = 40
# for i in range(4):
#     print(i)
#     for s in range(4):
#         print(seq_items_idxs[s])
#         plt.imshow(compressed_moments_sequences[i][idx,s])
#         plt.show()


# compressed_moments_3_sequences[1]


def get_kmeans_elbow_stds(x,max_n_clusters=20,verbose=1):
    """
        x: should be normalized per channel with normalize_channels_averages and flattened(1), shape: [N,C*H*W]
        returns an np.array of average of all standard deviations per value of each cluster  
    """
    clusters_stds = []
    for n_clusters in tqdm(range(1,max_n_clusters+1)):
        print(f"Calculating for {n_clusters} clusters...")
        clusters_dict = get_kmeans_clusters_dict(x,x,n_clusters,verbose=verbose)
        clusters_stds+=[sum([clusters_dict[i].std(0).sum() for i in range(n_clusters)])/n_clusters]
        print(f"Sum of standard deviation of clusters: {clusters_stds[-1]:.2f}")
    clusters_stds = np.array(clusters_stds)
    title = "Averages of K-Means Clusters' Standard Deviations"
    plt.clf();
    fig,axes=plt.subplots(1,1,figsize=(9,9),dpi=70);
    axes.title.set_text(title)
    axes.plot(np.arange(1,max_n_clusters+1),clusters_stds)
    axes.set_xlabel("Number of Clusters")    
    axes.set_ylabel("Average Sum of Clusters' Standard Deviation") 
    plt.show()
    return clusters_stds











for i,t in enumerate(compressed_moments_sequences):
    print(f"\n\n#### Compression {i}:")
    t_normed,_,_ = normalize_channels_averages(t.transpose(1,2))
    t_flat = t_normed.flatten(1)
    get_kmeans_elbow_stds(t_flat,max_n_clusters=20,verbose=1)








from sklearn.cluster import MiniBatchKMeans,KMeans


kmeans = MiniBatchKMeans(n_clusters=20)
kmeans.fit_predict(t_flat)


kmeans.score(t_flat)


def get_kmeans_clusters_dict_sklearn(x,x_raw,num_clusters,verbose=1,speed="fast"):
    """
        x: should be normalized per channel with normalize_channels_averages and flattened(1), shape: [N,C*H*W]
        x_raw: will be separated into clusters, original data
        returns a dict with {i: x[labels[i]}
    """
    kmeans = MiniBatchKMeans(n_clusters=num_clusters) if speed=="fast" else KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(x)
    score = kmeans.score(x)
    x_clustered_dict = {i: x_raw[labels==i] for i in range(num_clusters)}
    idxs_dict = {i: labels==i for i in range(num_clusters)}
    if verbose>0:
        num_all_labels = x.shape[0]
        for i in range(num_clusters):
            num_for_label = x_clustered_dict[i].shape[0]
            print(f"{i} : {x_clustered_dict[i].shape}, part from all labels: {100*num_for_label/num_all_labels:.2f}%")
    return x_clustered_dict, score, idxs_dict

def get_kmeans_elbow_scores_sklearn(x,max_n_clusters=20,verbose=1,speed="fast"):
    """
        x: should be normalized per channel with normalize_channels_averages and flattened(1), shape: [N,C*H*W]
        returns an np.array of average of all standard deviations per value of each cluster  
    """
    clusters_scores = []
    for n_clusters in tqdm(range(1,max_n_clusters+1)):
        print(f"Calculating for {n_clusters} clusters...")
        clusters_dict,score,_ = get_kmeans_clusters_dict_sklearn(x,x,n_clusters,verbose=verbose,speed=speed)
#         clusters_stds+=[sum([clusters_dict[i].std(0).sum() for i in range(n_clusters)])/n_clusters]
        clusters_scores+=[score]
        print(f"Sum of standard deviation of clusters: {clusters_scores[-1]:.2f}")
    clusters_scores = np.array(clusters_scores)
    title = "Averages of K-Means Clusters' Standard Deviations"
    plt.clf();
    fig,axes=plt.subplots(1,1,figsize=(9,9),dpi=70);
    axes.title.set_text(title)
    axes.plot(np.arange(1,max_n_clusters+1),clusters_scores)
    axes.set_xlabel("Number of Clusters")    
    axes.set_ylabel("Average Sum of Clusters' Standard Deviation") 
    plt.show()
    return clusters_scores


for i,t in enumerate(compressed_moments_sequences):
    print(f"\n\n#### Compression {i}:")
    t_normed,_,_ = normalize_channels_averages(t.transpose(1,2))
    t_flat = t_normed.flatten(1)
    get_kmeans_elbow_scores_sklearn(t_flat,max_n_clusters=20,verbose=1,speed="slow")




















num_clusters = 6
cluster_compression = compressed_moments_sequences[-1]
t_normed,_,_ = normalize_channels_averages(cluster_compression.transpose(1,2))
t_flat = t_normed.flatten(1)
clusters,_,idxs_dict = get_kmeans_clusters_dict_sklearn(t_flat,cluster_compression,
                                            num_clusters,verbose=1,speed="slow")


idxs_dict[5].shape


events_with_sequences = inputs_minus_seasonal[events_idxs_that_have_sequences]
events_with_sequences.shape


clusters_sequence_avg = {}
for i in range(num_clusters):
    cluster = events_with_sequences[idxs_dict[i]]
    size_cluster = cluster.shape[0]
    clusters_sequence_avg[i] = seq_handler.get_sequence(inputs_minus_seasonal,0,add_dim=True)*0
    for idx,event_idx in enumerate(events_idxs_that_have_sequences):
        if idxs_dict[i][idx]:
            clusters_sequence_avg[i]+=seq_handler.get_sequence(inputs_minus_seasonal,idx,add_dim=True)/size_cluster
    print(i,clusters_sequence_avg[i].shape,size_cluster)


print(f"{num_clusters}-Clustered Events")
channels_titles = [description["input"][i]["short"] for i in range(20)]
titles = [f"{i} Days Before Event, Cluster {cluster}" for cluster in range(6) for i in range(-6,1,2)]

for c in range(20):
    title = channels_titles[c]
    print("\n\nChannel: ",title)
    tensors = [clusters_sequence_avg[cluster][seq_i,c] for cluster in range(6) for seq_i in range(4)]
    print_tensors_with_cartopy(tensors, main_title=title, titles=titles,
                               lons=None, lats=None, save_as="", lock_bar=True, lock_bar_idxs=None,
                               num_levels=10)    







