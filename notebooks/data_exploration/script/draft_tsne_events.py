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
# from tsne_torch import TorchTSNE as TSNE #!pip install tsne-torch # too heavy for GPU....
from sklearn.manifold import TSNE


# !pip3 install tsne-torch


years_list = list(range(2003,2019))
data_dir = "../../data/datasets_20_81_189_3h_7days_future"
data_dir_compressed = data_dir+"/compressed"
base_filename = "dataset_20_81_189_3h_7days_future"
description = torch.load(f"{data_dir}/metadata/dataset_20_81_189_3h_7days_future_2003_descriptions.pkl")[0]
titles_channels = [description["input"][i]["long"] for i in range(20)]
event_threshold = 73.4

sequences_names = [
    "4days_light",
    "6days_light",
    "4days_heavy",
    "6days_heavy",
]


sequence_name = sequences_names[0]
base_filename_sequence_str = f"{data_dir}/{base_filename}_events_sequences_{sequence_name}"

events_sequences_inputs = torch.load(f"{base_filename_sequence_str}_inputs.pkl")
events_sequences_targets = torch.load(f"{base_filename_sequence_str}_targets.pkl")
events_sequences_timestamps = torch.load(f"{base_filename_sequence_str}_timestamps.pkl")
events_sequences_inputs.shape, events_sequences_targets.shape, len(events_sequences_timestamps)


avgs = events_sequences_inputs.mean([0,1,3,4])
stds = events_sequences_inputs.std([0,1,3,4])
events_sequences_inputs-=avgs[None,None,:,None,None]
events_sequences_inputs/=stds[None,None,:,None,None]








X = events_sequences_inputs.flatten(1)  # shape (n_samples, d)
X_emb = X_embedded = TSNE(n_components=2, learning_rate='auto',
                          init='random').fit_transform(X)


X_emb.shape


plt.scatter(X_emb[:,0],X_emb[:,1])


X_emb = TSNE(n_components=2, learning_rate='auto',
             init='random',n_iter=5000).fit_transform(X)
plt.scatter(X_emb[:,0],X_emb[:,1])


# average channels first...?


channels_to_average = [np.arange(1),
                       np.array([4,5,6,18]),np.array([7,8,9,19]),
                       np.arange(10,14),np.array([14])]

print(len(channels_to_average),channels_to_average)
for c in range(len(channels_to_average)):
    print(f"{c}:\n   {[description['input'][i]['short'] for i in channels_to_average[c]]}")


events_sequences_inputs_averaged_channels = []
for channels in channels_to_average:
    events_sequences_inputs_averaged_channels.append(events_sequences_inputs[:,:,channels,:,:].mean([2]))
events_sequences_inputs_averaged_channels = torch.stack(events_sequences_inputs_averaged_channels,dim=2)
events_sequences_inputs_averaged_channels.shape


X = events_sequences_inputs_averaged_channels.flatten(1)
X_emb = TSNE(n_components=2, learning_rate='auto',
                          init='random',n_iter=5000).fit_transform(X)
plt.scatter(X_emb[:,0],X_emb[:,1])


X = events_sequences_inputs.flatten(1)  # shape (n_samples, d)
X_emb = TSNE(n_components=2, learning_rate='auto',
                          init='random',n_iter=10000).fit_transform(X)
plt.scatter(X_emb[:,0],X_emb[:,1])


X = events_sequences_inputs_averaged_channels.flatten(1)
X_emb = TSNE(n_components=2, learning_rate='auto',
                          init='random',n_iter=10000).fit_transform(X)
plt.scatter(X_emb[:,0],X_emb[:,1])








# compressed_moments_7 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_7.pkl")
compressed_sift_pca_50 = torch.load(f"{data_dir_compressed}/{base_filename}_inputs_compressed_moments_8.pkl")
compressed_sift_pca_50.shape


timestamps_all = timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
len(timestamps_all)


idxs_of_events_with_sequences = []
for t in timestamps:
    if


89*189*20




