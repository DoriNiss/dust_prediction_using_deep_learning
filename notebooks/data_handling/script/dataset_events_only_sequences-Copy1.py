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


events_bool_idxs_original = targets[:,0]>=event_threshold
events_idxs_full_original_list = np.arange(len(events_bool_idxs_original))[events_bool_idxs_original]


seq_items_idxs_4_days_light = [int(i) for i in [-4*8,-3*8,-2*8,-1*8,0]]
seq_items_idxs_6_days_light = [int(i) for i in [-6*8,-5*8,-4*8,-3*8,-2*8,-1*8,0]]
seq_items_idxs_4_days_heavy = [int(i) for i in [-4*8,-3*8,-2.5*8,-2*8,-1.5*8,-1*8,-6,-4,-2,0]]
seq_items_idxs_6_days_heavy = [int(i) for i in [-6*8,-5*8,-4*8,-3*8,-2.5*8,-2*8,-1.5*8,-1*8,-6,-4,-2,0]]

sequences_items = [
    seq_items_idxs_4_days_light,
    seq_items_idxs_6_days_light,
    seq_items_idxs_4_days_heavy,
    seq_items_idxs_6_days_heavy,
]
save_as_strs = [
    "4days_light",
    "6days_light",
    "4days_heavy",
    "6days_heavy",
]

num_sequences=4


def original_idxs_of_events_with_sequences(sequence_handler):
    idxs_events_handler = [sequence_handler.translate_original_idx_to_handler(idx)
                           for idx in events_idxs_full_original_list]
    idxs_events_handler = [idx for idx in idxs_events_handler if idx is not None]
    idxs_events_original_with_sequences = [sequence_handler.translate_handler_idx_to_original(idx)
                                           for idx in idxs_events_handler]
    return idxs_events_original_with_sequences


for i in tqdm(range(num_sequences)):
    seq_items = sequences_items[i]
    save_as_str = f"{data_dir}/{base_filename}_events_sequences_{save_as_strs[i]}_"
    print(f"\n\n#### {i}: Creating event sequences {save_as_strs[i]}: {len(seq_items)}:{seq_items}\n")
    seq_handler = SequentialHandler(timestamps,seq_items,timesteps=[3,"h"],verbose_level=1)
    print("... Calculating original indices of events that have sequences...")
    original_events_idxs = original_idxs_of_events_with_sequences(seq_handler)
    print(f"... Done! {len(original_events_idxs)}")
    print(f"... Building timestamps...")
    timestamps_events = timestamps[original_events_idxs]
    torch.save(timestamps_events,save_as_str+"timestamps.pkl")
    print(f"... Done! {len(timestamps_events)},{timestamps_events}, saved as {save_as_str}timestamps.pkl")
    print(f"... Building targets...")
    targets_events = seq_handler.get_batched_sequences_from_original_idxs(targets,original_events_idxs)
    torch.save(targets_events,save_as_str+"targets.pkl")
    print(f"... Done! {targets_events.shape}, saved as {save_as_str}targets.pkl")
    print(f"... Building inputs...")
    inputs_events = seq_handler.get_batched_sequences_from_original_idxs(inputs_minus_avgs,original_events_idxs)
    torch.save(inputs_events,save_as_str+"inputs.pkl")
    print(f"... Done! {inputs_events.shape}, saved as {save_as_str}inputs.pkl")
    



















