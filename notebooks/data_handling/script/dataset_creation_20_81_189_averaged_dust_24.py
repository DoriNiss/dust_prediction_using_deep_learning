#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/')
from utils.files_loading import *
from data_handlers.tensors_labeling_and_averaging import *
from tqdm import tqdm


data_dir_original = "../../data/datasets_20_81_189"
data_dir_new = "../../data/datasets_20_81_189_averaged_dust_24h"
base_filename_original = "dataset_20_81_189"
base_filename_new = "dataset_20_81_189_averaged_dust_24h"
description_new_path = data_dir_new+"/metadata/"+base_filename_new+"_metadata.pkl"
suffix_target = "target.pkl"
suffix_input = "input.pkl"
suffix_timestamps = "timestamps.pkl"
years_list = list(range(2003,2019))


description_original = torch.load("../../data/meteorology_dataframes_23_81_189/metadata/meteorology_dataframes_20_81_189_description.pkl")
# description_original["target"]


description = description_original.copy()
description["target"] = {}
description["target"][0] = "dust at i+0h"
for i in range(1,8):
    hours = [str(6*h)+"h" for h in range((i-1)*4,i*4)]
    description["target"][i] = f"average of dust of {hours} (average of the following {(i-1)*24} to {i*24} hours)"
for i in range(8,15):
    hours = [str(-h)+"h" for h in range(24*(15-i),24*(14-i),-6)]
    description["target"][i] = f"average of dust of {hours} (average of the last {24*(15-i)} to {24*(14-i)} hours)"
description["target"][15] = "delta dust at i+0h"
for i in range(1,8):
    hours = [str(6*h)+"h" for h in range((i-1)*4,i*4)]
    description["target"][i+15] = f"average of delta dust of {hours} (average of the following {(i-1)*24} to {i*24} hours)"
for i in range(8,15):
    hours = [str(-h)+"h" for h in range(24*(15-i),24*(14-i),-6)]
    description["target"][i+15] = f"average of delta dust of {hours} (average of the last {24*(15-i)} to {24*(14-i)} hours)"
description["target"][30] = "Label: threshold = 73.4[ug/m3], label=0 if dust_0<threshold, label=1 if dust_0>=threshold"
description["target"]


# cols after duplicating col 0 at [0] (dust 0) and col 57 (which turns to 58) at [58] (delta_0):
cols_to_avg_dust_0 = [np.array([0])]
valid_threshold_dust_0 = [0]
cols_to_avg_delta_0 = [np.array([58])]
valid_threshold_delta_0 = [0]

cols_to_avg_dust_future = [np.arange(i,i+4) for i in range(1,29,4)] #ignoring 168h (29 in the col_0 duplicated tensor)
cols_to_avg_dust_past = [np.arange(i,i+4) for i in range(30,58,4)]
cols_to_avg_delta_dust_future = [np.arange(i,i+4) for i in range(59,86,4)]
cols_to_avg_delta_dust_past = [np.arange(i,i+4) for i in range(88,116,4)]

valid_thresholds = [0.5]*14

cols_to_average = (cols_to_avg_dust_0+cols_to_avg_dust_future+cols_to_avg_dust_past+
                   cols_to_avg_delta_0+cols_to_avg_delta_dust_future+cols_to_avg_delta_dust_past)

valid_threshold = valid_threshold_dust_0+valid_thresholds+valid_threshold_delta_0+valid_thresholds


for i in range(len(cols_to_average)):
    print(f"\n{i}: cols: {cols_to_average[i]}, th: {valid_threshold[i]}")
    print(description["target"][i])


# description_original["target"]


# torch.save(description,description_new_path)
# torch.load(description_new_path)


thresholds = [73.4]
labels = [0,1]


for y in tqdm(years_list):
    print(f"\nYear {y}:")
    original_target = torch.load(f"{data_dir_original}/{base_filename_original}_{y}_{suffix_target}")
    original_input = torch.load(f"{data_dir_original}/{base_filename_original}_{y}_{suffix_input}")
    original_timestamps = torch.load(f"{data_dir_original}/{base_filename_original}_{y}_{suffix_timestamps}")
    new_target = duplicate_col_i(original_target,0)
    new_target = duplicate_col_i(new_target,58)
    new_target = add_labels(new_target,thresholds=thresholds,labels=labels,label_by_col=0)
    new_target,rows_to_keep = average_cols_and_drop_invalid(new_target,cols_to_average,valid_threshold,
                                                            invalid_values=[0, np.nan])
    print(f"Original length: {len(original_timestamps)}, new length: {len(rows_to_keep)}")
    new_target = new_target[rows_to_keep]
    new_input = original_input[rows_to_keep]
    new_timestamps = original_timestamps[rows_to_keep]
    torch.save(new_target,f"{data_dir_new}/{base_filename_new}_{y}_{suffix_target}")
    torch.save(new_input,f"{data_dir_new}/{base_filename_new}_{y}_{suffix_input}")
    torch.save(new_timestamps,f"{data_dir_new}/{base_filename_new}_{y}_{suffix_timestamps}")
    


sample_year = 2015

sample_input = torch.load(f"{data_dir_new}/{base_filename_new}_{sample_year}_{suffix_input}")
sample_target = torch.load(f"{data_dir_new}/{base_filename_new}_{sample_year}_{suffix_target}")
sample_timestamps = torch.load(f"{data_dir_new}/{base_filename_new}_{sample_year}_{suffix_timestamps}")

print(sample_input.shape,sample_target.shape,len(sample_timestamps))










