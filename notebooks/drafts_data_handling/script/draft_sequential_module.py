#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from SequentialHandler import *


years_list = [2003,2004,2005]
paths_dir = "../../data/datasets_20_81_189" 
base_filename = "dataset_20_81_189"
suffix_input = "input"
suffix_target = "target"
suffix_timestamps = "timestamps"
paths_inputs = SequentialHandler.get_paths_from_years_list(years_list, paths_dir, base_filename, suffix_input)
paths_targets = SequentialHandler.get_paths_from_years_list(years_list, paths_dir, base_filename, suffix_target)
paths_timestamps = SequentialHandler.get_paths_from_years_list(years_list, paths_dir, base_filename, suffix_timestamps)

paths_inputs ,paths_targets,paths_timestamps


t = SequentialHandler.get_stacked_tensor_from_paths(paths_inputs)


handler = SequentialHandler.get_handler_from_timestamps_paths(paths_timestamps,[0,-8,-4])


t_sample = handler.get_sequence(t,4)
t_sample.shape


t_sample = handler.get_sequence(t,4,add_dim=False)
t_sample.shape


t_all = handler.get_all_sequences(t)
t_all.shape


t_all = handler.get_all_sequences(t,add_dim=False)
t_all.shape











years_list = [y for y in range(2006,2010)]

paths_dir = "../../data/datasets_20_81_189" 
base_filename = "dataset_20_81_189"
suffix_timestamps = "timestamps"
paths_timestamps = SequentialHandler.get_paths_from_years_list(years_list, paths_dir, base_filename, suffix_timestamps)
paths_timestamps


sequence_items = list(range(-2*6,1))
print(sequence_items)


handler = SequentialHandler.get_handler_from_timestamps_paths(paths_timestamps,sequence_items)


suffix_input = "input"
paths_inputs = SequentialHandler.get_paths_from_years_list(years_list, paths_dir, base_filename, suffix_input)
paths_inputs


t = SequentialHandler.get_stacked_tensor_from_paths(paths_inputs)


idx_sample = 10
handler.print_sample(idx_sample)





seq_sample = handler.get_sequence(t,idx_sample)
seq_sample.shape


seq_sample = handler.get_sequence(t,idx_sample,add_dim=False)
seq_sample.shape


# seq_all = handler.get_all_sequences(t)
# seq_all.shape




