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





# ! ls ../../data/dust_MOEP_2019_2020


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


55520+5848


new_dust_dataframe_debug = new_dust_dataframe[:10]
new_dust_dataframe_debug


new_dust_dataframe_debug["BEER_SHEVA_PM10_0"]


torch.save(new_dust_dataframe_debug,new_dust_dataframe_debug_filename)


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


dims_cols_strings = {1:all_stations, 2:pm_types, 3:data_suffixes}
dims_cols_strings





df = new_dust_dataframe_debug
data_handler_dust = DatasetHandler_DataframeToTensor_Dust(    
    df, dims_cols_strings, metadata=metadata, timestamps=None, 
    save_base_filename=result_tensor_base_filename_debug, invalid_col_fill=-7777, save_timestamps=True,
)
data_handler_dust.create_yearly_datasets([2019,2020], add_years_to_name=True)





t_debug = torch.load(f"{result_dir}/dust_61368_108_2_339_debug_2019_tensor.pkl")
timestamps_debug = torch.load(f"{result_dir}/dust_61368_108_2_339_debug_2019_tensor.pkl")
t_debug.shape,len(timestamps_debug)


t_debug


new_dust_dataframe_debug["REHOVOT_PM10_0"],new_dust_dataframe_debug["REHOVOT_PM25_24"],new_dust_dataframe_debug["REHOVOT_PM10_values_count_48"]


t_debug


dims_cols_strings[1],dims_cols_strings[1][21]


dims_cols_strings[3],dims_cols_strings[3][-41]


t_debug.shape


t_debug[:,21,0,0],t_debug[:,21,1,64],t_debug[:,21,0,-41]


new_dust_dataframe_debug["REHOVOT_PM10_0"],new_dust_dataframe_debug["REHOVOT_PM25_24"],new_dust_dataframe_debug["REHOVOT_PM10_values_count_48"]




