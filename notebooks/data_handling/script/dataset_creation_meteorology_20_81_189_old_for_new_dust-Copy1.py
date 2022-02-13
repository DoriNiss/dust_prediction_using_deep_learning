#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import csv
from tqdm import tqdm
from joblib import Parallel, delayed #conda install -c anaconda joblib

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DatasetHandler_DataframeToTensor_Meteorology import *


data_dir = "../../data"
meteorology_dataframe_dir = f"{data_dir}/meteorology_dataframes_20_81_189_3h"
result_dir = f"{data_dir}/dataset_input_20_81_189_old_dust_108_2_339"
file_basename = f"{result_dir}/meteorology_20_81_189_old"
metadata_filename = f"{data_dir}/meteorology_dataframes_20_81_189_3h/metadata/meteorology_dataframes_20_81_189_3h_description.pkl"
timestamps_dir = f"{data_dir}/dust_55520_108_2_339"


metadata_all = torch.load(metadata_filename)["input"]
metadata_all


old_format_channels_dict = {
    "SLP": {"tensor_channels":np.array([0]), "df_channels":np.array([0])},
    "Z": {"tensor_channels":np.array([1,2,3]), "df_channels":np.array([0,1,2])},
    "U": {"tensor_channels":np.array([4,5,6]), "df_channels":np.array([0,1,2])},
    "V": {"tensor_channels":np.array([7,8,9]), "df_channels":np.array([0,1,2])},
    "PV": {"tensor_channels":np.array([10,11,12,13]), "df_channels":np.array([0,1,2,3])},
    "aod550": {"tensor_channels":np.array([14]), "df_channels":np.array([0])},
    "duaod550": {"tensor_channels":np.array([15]), "df_channels":np.array([0])},
    "aermssdul": {"tensor_channels":np.array([16]), "df_channels":np.array([0])},
    "aermssdum": {"tensor_channels":np.array([17]), "df_channels":np.array([0])},
    "u10": {"tensor_channels":np.array([18]), "df_channels":np.array([0])},
    "v10": {"tensor_channels":np.array([19]), "df_channels":np.array([0])},
}


dims_cols_strings = {1:[c_str for c_str in old_format_channels_dict.keys()]}
dims_cols_strings


lons,lats = np.arange(20,60.5,0.5),np.arange(-44,50.5,0.5)
len(lons),len(lats)





def load_and_save_year(year):
    print(f"\n\n####### YEAR {year}:")
    meteorology_dataframe_filename = f"{meteorology_dataframe_dir}/meteorology_dataframe_20_81_189_3h_{year}.pkl"
    timestamps_filename = f"{timestamps_dir}/dust_dataset_{year}_timestamps.pkl"
    df = torch.load(meteorology_dataframe_filename)
    file_year_basename = f"{file_basename}_{year}"
    timestamps = torch.load(timestamps_filename)
    handler = DatasetHandler_DataframeToTensor_Meteorology(
        df, dims_cols_strings=dims_cols_strings, metadata=metadata_all, timestamps=timestamps, 
        save_base_filename=file_year_basename,
        invalid_col_fill=-777, param_shape=[81,189], lons=lons, lats=lats, save_timestamps=True,
        old_format_channels_dict=old_format_channels_dict
    )
    handler.create_yearly_datasets([year])
    


years = list(range(2003,2019))
Parallel(n_jobs=1,verbose=100)(delayed(load_and_save_year)(year)
        for year in years
)    
    





# data_dir = "../../data"
# meteorology_dataframe_dir = f"{data_dir}/meteorology_dataframes_20_81_189_3h"
# result_dir = f"{data_dir}/dataset_input_20_81_189_old_dust_108_2_339"
# file_basename = f"{result_dir}/meteorology_20_81_189_old"
# metadata_filename = f"{data_dir}/meteorology_dataframes_20_81_189_3h/metadata/meteorology_dataframes_20_81_189_3h_description.pkl"
# timestamps_dir = f"{data_dir}/dust_55520_108_2_339"


tensors_list,timestamps_lists=[],[]
for year in years:
    tensors_list.append(torch.load(f"{file_basename}_{year}_{year}_tensor.pkl"))
    timestamps_lists.append(torch.load(f"{file_basename}_{year}_{year}_timestamps.pkl"))
#     timestamps_lists.append(torch.load(f"{timestamps_dir}/dust_dataset_{year}_timestamps.pkl"))

full_tensor, full_timestamps = DatasetHandler_DataframeToTensor_Meteorology.merge_by_timestamps(
    tensors_list, timestamps_lists)

print(full_tensor.shape, len(full_timestamps))

torch.save(full_tensor,f"{file_basename}_tensor_full.pkl")
torch.save(full_timestamps,f"{file_basename}_timestamps_full.pkl")


f"{file_basename}_tensor_full.pkl",f"{file_basename}_timestamps_full.pkl"


#46736

