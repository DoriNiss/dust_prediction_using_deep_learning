#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import csv
from tqdm import tqdm

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DatasetHandler_DataframeToTensor_Meteorology import *


data_dir = "../../data"
meteorology_dataframe_debug_dir = f"{data_dir}/meteorology_dataframes_20_81_189_3h/debug"
meteorology_dataframe_filename_debug = f"{meteorology_dataframe_debug_dir}/meteorology_dataframe_20_81_189_3h_debug_2003.pkl"
result_dir_debug = f"{data_dir}/dataset_input_20_81_189_old_dust_108_2_339/debug"
file_basename = f"{result_dir_debug}/meteorology_20_81_189_old_debug"
metadata_filename = f"{data_dir}/meteorology_dataframes_20_81_189_3h/metadata/meteorology_dataframes_20_81_189_3h_description.pkl"


df_debug = torch.load(meteorology_dataframe_filename_debug)


df_debug["aod550"][0].shape


metadata_df = torch.load(metadata_filename)


# metadata_df = metadata_df["input"]
metadata_df


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


dims_cols_strings = {1:[c_str for c_str in df_debug.columns]}


dims_cols_strings


lons,lats = np.arange(20,60.5,0.5),np.arange(-44,50.5,0.5)
len(lons),len(lats)


handler = DatasetHandler_DataframeToTensor_Meteorology(
    df_debug, dims_cols_strings=dims_cols_strings, metadata={}, timestamps=None, save_base_filename=file_basename,
    invalid_col_fill=-777, param_shape=[81,189], lons=lons, lats=lats, save_timestamps=False,
    old_format_channels_dict=old_format_channels_dict
)





handler.create_yearly_datasets([2003])


x_debug_2003 = torch.load(f"{file_basename}_2003_tensor.pkl")
metadata_debug_2003 = torch.load(f"{file_basename}_metadata.pkl")


x_debug_2003.shape


metadata_debug_2003




