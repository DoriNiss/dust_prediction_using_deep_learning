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
dust_dataframe_base_filename = f"{data_dir}/dust_multistations/dust_df_all_stations_2000_to_2018_pm10_pm25_past7d_future7d"
dataset_dir = f"{data_dir}/dataset_input_20_81_189_old_dust_108_2_339"
file_basename = f"{dataset_dir}/dust_108_2_339"


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


for year in range(2000,2019):
    df = torch.load(f"{dust_dataframe_base_filename}_full_cols_{year}.pkl")
    print(f"\n######## Loaded year {year}: {len(df)}")
    data_handler_dust = DatasetHandler_DataframeToTensor_Dust(    
        df, dims_cols_strings, metadata=metadata, timestamps=None, save_base_filename=f"{file_basename}",
        invalid_col_fill=-777, save_timestamps=True,
    )
    data_handler_dust.create_yearly_datasets([year], add_years_to_name=True)


years = list(range(2000,2019))
DatasetHandler_DataframeToTensor_Dust.load_merge_and_save_yearly_tensors_by_timestamps(
    file_basename, years ,metadata=data_handler_dust.metadata
)











x = torch.load(f"{file_basename}_merged_tensor.pkl")
x.shape


merge_timestamps = torch.load(f"{file_basename}_merged_timestamps.pkl")


len(merge_timestamps)


result_metadata = torch.load(f"{file_basename}_merged_metadata.pkl")
result_metadata







