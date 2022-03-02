#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import csv
from tqdm import tqdm

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DatasetHandler_DataframeToTensor_Dust import *


data_dir = "../../data"
metadata_base_filename = f"{data_dir}/dust_multistations/metadata/dust_df_all_stations_2000_to_2018_pm10_pm25_past7d_future7d_metadata"
metadata_all_filename = f"{metadata_base_filename}_all.pkl"
dust_dataframe_base_filename = f"{data_dir}/dust_MOEP_2019_2020"
old_dust_dir = f"{data_dir}/dataset_input_20_81_189_old_dust_108_2_339"
old_file_basename = f"{old_dust_dir}/dust_108_2_339"
old_metadata = f"{old_dust_dir}/dust_108_2_339_merged_metadata.pkl"
old_tensor = f"{old_dust_dir}/dust_108_2_339_merged_tensor.pkl"
old_timestamps = f"{old_dust_dir}/dust_108_2_339_merged_timestamps.pkl"

new_dust_dir = f"{data_dir}/dust_MOEP_2019_2020"
new_dust_dataframe_filename = f"{new_dust_dir}/dust_df_MOEP_2019_2020_full.pkl"
new_dust_dataframe_debug_filename = f"{new_dust_dir}/dust_df_MOEP_2019_2020_full_debug.pkl"

result_dir = f"{data_dir}/dust_61368_108_2_339"
result_tensor_base_filename = f"{result_dir}/dust_61368_108_2_339"
result_tensor_base_filename_debug = f"{result_dir}/dust_61368_108_2_339_debug"


new_dust_dataframe = torch.load(new_dust_dataframe_filename)
len(new_dust_dataframe)


metadata = torch.load(metadata_all_filename)
all_stations = [station["Name"] for station in metadata["all"]]
print(f"All stations: ({len(all_stations)})\n{all_stations}")

pm_types = ["PM10","PM25"]

lags = [0]+[i for i in range(-7*24,0,3)]+[i for i in range(3,7*24+3,3)]
lag_suffixes = []
for lag in lags:
    lag_suffix = f"{lag}" if lag>=0 else f"m{-lag}"
    lag_suffixes.append(lag_suffix)
data_suffixes = lag_suffixes+[f"delta_{lag_str}" for lag_str in lag_suffixes]+                [f"values_count_{lag_str}" for lag_str in lag_suffixes]
print(f"lags: {lags}, {data_suffixes}, {len(data_suffixes)}")


metadata


dims_cols_strings = {1:all_stations, 2:pm_types, 3:data_suffixes}
dims_cols_strings


df = new_dust_dataframe
data_handler_dust = DatasetHandler_DataframeToTensor_Dust(    
    df, dims_cols_strings, metadata=metadata, timestamps=None, 
    save_base_filename=result_tensor_base_filename, invalid_col_fill=-7777, save_timestamps=True,
)
data_handler_dust.create_yearly_datasets([2019,2020], add_years_to_name=True)


metadata_new_years = torch.load(f"{result_dir}/dust_61368_108_2_339_metadata.pkl")
metadata_new_years


tensors_list, timestamps_list = [],[]
tensors_path_list = [
    old_tensor,
    f"{result_dir}/dust_61368_108_2_339_2019_tensor.pkl",
    f"{result_dir}/dust_61368_108_2_339_2020_tensor.pkl"
]

timestamps_path_list = [
    old_timestamps,
    f"{result_dir}/dust_61368_108_2_339_2019_timestamps.pkl",
    f"{result_dir}/dust_61368_108_2_339_2020_timestamps.pkl",
]

for path_idx in tqdm(range(len(tensors_path_list))):
    tensors_list.append(torch.load(tensors_path_list[path_idx]))
    timestamps_list.append(torch.load(timestamps_path_list[path_idx]))
    


for i in range(len(tensors_path_list)):
    print(i,tensors_list[i].shape,len(timestamps_list))


merged_tensor,merged_timestamps = DatasetHandler_DataframeToTensor_Dust.merge_by_timestamps(
    tensors_list, timestamps_list
)
print(f"Merged! {merged_tensor.shape}, {len(merged_timestamps)}")


merged_timestamps


torch.save(merged_tensor,f"{result_dir}/dust_61368_108_2_339_full_tensor.pkl")
torch.save(merged_timestamps,f"{result_dir}/dust_61368_108_2_339_full_timestamps.pkl")




