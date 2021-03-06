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
dust_dataframe_filename = f"{data_dir}/dust_multistations/dust_df_all_stations_2000_to_2018_pm10_pm25_past7d_future7d.pkl"
dust_dataset_dir = f"{data_dir}/dust_55520_108_2_339"
file_basename = f"{dust_dataset_dir}/dust_dataset"


metadata = torch.load(metadata_all_filename)
all_stations = [station["Name"] for station in metadata["all"]]
print(f"All stations: ({len(all_stations)})\n{all_stations}")

pm_types = ["PM10","PM25"]

lags = [0]+[i*24 for i in range(-7,0)]+[i*24 for i in range(1,8)]
lag_suffixes = []
for lag in lags:
    lag_suffix = f"{lag}" if lag>=0 else f"m{-lag}"
    lag_suffixes.append(lag_suffix)
data_suffixes = lag_suffixes+[f"delta_{lag_str}" for lag_str in lag_suffixes]+                [f"values_count_{lag_str}" for lag_str in lag_suffixes]
print(f"lags: {lags}, {data_suffixes}, {len(data_suffixes)}")


dims_cols_strings = {1:all_stations, 2:pm_types, 3:data_suffixes}
dims_cols_strings


dust_dataframe = torch.load(dust_dataframe_filename)


len(dust_dataframe)


data_handler_dust = DatasetHandler_DataframeToTensor_Dust(    
    dust_dataframe, dims_cols_strings, metadata=metadata, timestamps=None, save_base_filename=file_basename,
    invalid_col_fill=-777
)


years=[y for y in range(2000,2021)]

data_handler_dust.create_yearly_datasets_parallel(years,njobs=3)


DatasetHandler_DataframeToTensor_Dust.load_merge_and_save_yearly_tensors_by_timestamps(
    file_basename, years ,metadata=data_handler_dust.metadata
)








original_timestamps = dust_dataframe.index
result_timestamps = data_handler_dust.timestamps


len(data_handler_dust.dataframe), len(result_timestamps), len(original_timestamps)


merge_timestamps = torch.load(f"{file_basename}_merged_timestamps.pkl")


len(merge_timestamps)


for result_timestamp in tqdm(original_timestamps):
    if result_timestamp not in merge_timestamps:
        print(result_timestamp)











result_metadata = torch.load(f"{file_basename}_merged_metadata.pkl")


# result_metadata["idxs"]["dims"][0] = \
# f'timestamps, {merge_timestamps[0]} to 2018-12-31 21:00:00+00:00, len={len(merge_timestamps)}'


# result_metadata["idxs"]["dims"][0]


result_metadata







