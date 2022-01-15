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
dust_dataset_dir = f"{data_dir}/dust_55521_108_2_45"
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




