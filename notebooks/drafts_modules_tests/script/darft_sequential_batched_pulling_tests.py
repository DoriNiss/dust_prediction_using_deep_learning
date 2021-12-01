#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers/')
import torch
from SequentialHandler import *

data_dir = "../../data/datasets_20_81_189_averaged_dust_24h"
base_filename = "dataset_20_81_189_averaged_dust_24h"
targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
targets.shape,len(timestamps)


seq_items_idxs = [-6*4,-4*4,-2*4,0]


seq_handler = SequentialHandler(timestamps,seq_items_idxs,timesteps=[6,"h"],verbose_level=1)


sample_handler_idxs = [2,10]
sample = seq_handler.get_batched_sequences(targets,sample_handler_idxs)
sample.shape, [seq_handler.translate_handler_idx_to_original(i) for i in sample_handler_idxs]


sample[0], sample[1]


sample_from_original = seq_handler.get_batched_sequences_from_original_idxs(targets,[26, 34])
sample_from_original-sample





sys.path.insert(0, '../../packages/')
from utils.files_loading import *


inputs_sample,targets_sample,timestamps_sample = load_stacked_inputs_targets_timestamps_from_years_list([2005], 
                                                                                                        data_dir, 
                                                                                                        base_filename)


seq_handler = SequentialHandler(timestamps_sample,seq_items_idxs,timesteps=[6,"h"],verbose_level=1)


idxs_handler = [0,5,78]
idxs_original = [seq_handler.translate_handler_idx_to_original(i_handler) for i_handler in idxs_handler]
idxs_handler,idxs_original


inputs_dataset_handler,targets_dataset_handler,timestamps_dataset_handler = seq_handler.get_dataset_from_handler_idx(0,inputs_sample,targets_sample,timestamps_sample)
inputs_dataset_handler.shape,targets_dataset_handler.shape,timestamps_dataset_handler


seq_handler.rows_that_have_sequences[0]


inputs_dataset_original,targets_dataset_original,timestamps_dataset_original = seq_handler.get_dataset_from_original_idx(24,inputs_sample,targets_sample,timestamps_sample)
inputs_dataset_original.shape,targets_dataset_original.shape,timestamps_dataset_original


inputs_dataset_original-inputs_dataset_handler


inputs_dataset_handler,targets_dataset_handler,timestamps_dataset_handler = seq_handler.get_batched_dataset_from_handler_idxs(idxs_handler,inputs_sample,targets_sample,timestamps_sample)
inputs_dataset_handler.shape,targets_dataset_handler.shape,timestamps_dataset_handler


inputs_dataset_original,targets_dataset_original,timestamps_dataset_original = seq_handler.get_batched_dataset_from_original_idxs(idxs_original,inputs_sample,targets_sample,timestamps_sample)
inputs_dataset_original.shape,targets_dataset_original.shape,timestamps_dataset_original


inputs_dataset_handler-inputs_dataset_original


targets_dataset_original-targets_dataset_handler




