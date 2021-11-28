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




