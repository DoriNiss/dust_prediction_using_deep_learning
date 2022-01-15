#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import csv
from tqdm import tqdm

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DatasetHandler_DataframeToTensor_Super import *


data_dir = "../../data"
metadata_base_filename = f"{data_dir}/dust_multistations/metadata/dust_df_all_stations_2000_to_2018_pm10_pm25_past7d_future7d_metadata"
metadata_10_filename = f"{metadata_base_filename}_PM10.pkl"
metadata_25_filename = f"{metadata_base_filename}_PM25.pkl"
metadata_all_filename = f"{metadata_base_filename}_all.pkl"
debug_result_filename = f"{data_dir}/dust_multistations/debug_dataframe.pkl"


debug_dataframe = torch.load(debug_result_filename)


debug_dataframe


# !ls ../../data/dust_multistations/metadata





# metadata_10 = torch.load(metadata_10_filename)
# metadata_25 = torch.load(metadata_25_filename)
# metadata = {"PM10": metadata_10, "PM25": metadata_25}
# stations_10 = [station["Name"] for station in metadata_10[:-1]]
# stations_25 = [station["Name"] for station in metadata_25[:-1]]
# all_stations = list(set(stations_10)|set(stations_25))
# all_stations.remove("BEER_SHEVA")
# all_stations = ["BEER_SHEVA"]+all_stations
# metadata_all = []
# for station_name in all_stations:
#     if station_name in stations_10:
#         metadata_all.append(metadata_10[stations_10.index(station_name)])
#     elif station_name in stations_25:
#         metadata_all.append(metadata_25[stations_25.index(station_name)])
# metadata["all"] = metadata_all
# print(f"Stations with PM10: ({len(stations_10)})\n{stations_10}")
# print(f"Stations with PM2.5: ({len(stations_25)})\n{stations_25}")
# print(f"All stations: ({len(all_stations)})\n{all_stations}")
# torch.save(metadata,metadata_all_filename)

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
lags, data_suffixes, len(data_suffixes)


dims_cols_strings = {1:all_stations, 2:pm_types, 3:data_suffixes}








dust_dataset_dir_debug = f"{data_dir}/dust_108_2_45/debug"
file_debug_basename = f"{dust_dataset_dir_debug}/dust_dataset_debug"

tensor_handler = DatasetHandler_DataframeToTensor_Super(
    debug_dataframe, dims_cols_strings, metadata={}, timestamps=None, save_base_filename=file_debug_basename,
)


timestamps_test = debug_dataframe.index[:5]
tensor_test1 = tensor_handler.get_tensor_from_timestamps(timestamps_test)
tensor_test1.shape


print(tensor_test1[:,0,0,0])
print(tensor_test1[:,0,1,1])
print(tensor_test1[:,0,0,-15])
tensor_test1


print(debug_dataframe[:5]["BEER_SHEVA_PM10_0"])
print(debug_dataframe[:5]["BEER_SHEVA_PM25_m24"])
print(debug_dataframe[:5]["BEER_SHEVA_PM10_values_count_0"])


year = 2020
timestamps_test2 = debug_dataframe.index[debug_dataframe.index.year==year]
timestamps_test2.empty





years=[2006,2007,2020]
# tensor_handler.create_yearly_datasets(years) 


years=[2003,2004,2009,2020]

# tensor_handler.create_yearly_datasets_parallel(years,njobs=3)


tensor_handler.timestamps


f"{tensor_handler.save_base_filename}_{year}_timestamps.pkl"





# DatasetHandler_DataframeToTensor_Super.load_merge_and_save_yearly_tensors_by_timestamps(
#     file_debug_basename, [y for y in range(2005,2021)],metadata=metadata
# )








from data_handlers.DatasetHandler_DataframeToTensor_Dust import *


datahandler_dust = DatasetHandler_DataframeToTensor_Dust(    
    debug_dataframe, dims_cols_strings, metadata={}, timestamps=None, save_base_filename=None,
    invalid_col_fill=-777
)


datahandler_dust.metadata











dims_cols_strings_test = {
    1:["A","B","C"],
    2:["10","25"],
    3:["0","m24","24","48"]
}
dims_cols_strings_test


timestamps_test = debug_dataframe.index[:5]
df_test = pd.DataFrame({},index=timestamps_test)


last_dim = list(dims_cols_strings_test.keys())[-1]
all_col_names = []

def get_full_col_names(dim,name_so_far):
    if dim==last_dim:
        for col_str in dims_cols_strings_test[dim]:
            col_full_name = f"{name_so_far}{col_str}"
            all_col_names.append(col_full_name)
        return
    for col_str in dims_cols_strings_test[dim]:
        get_full_col_names(dim+1,name_so_far+f"{col_str}_")    

get_full_col_names(1,"")
all_col_names, len(all_col_names), 3*2*4

# for first_col_str in dims_cols_strings_test[1]:
#     for sec_col_str in dims_cols_strings_test[2]:
#         for thrd_col_str in dims_cols_strings_test[3]:
#             print(f"{first_col_str}_{sec_col_str}_{thrd_col_str}")


for i,col in enumerate(all_col_names):
    df_test[col] = [i*100+j for j in range(len(df_test))]
df_test


last_dim = list(dims_cols_strings_test.keys())[-1]
t = torch.zeros([len(timestamps_test)]+[len(dims_cols_strings_test[dim]) for dim in dims_cols_strings_test.keys()])
t.shape


def populate_tensor_from_col(t,dim,col_name_so_far):
    if dim==last_dim:
        for col_idx,col_str in enumerate(dims_cols_strings_test[dim]):
            col_full_name = f"{col_name_so_far}{col_str}"
            t[:,col_idx] +=df_test[col_full_name].values
        return
    for col_idx,col_str in enumerate(dims_cols_strings_test[dim]):
        populate_tensor_from_col(t[:,col_idx],dim+1,col_name_so_far+f"{col_str}_")

populate_tensor_from_col(t,1,"")
t


t_list = [t,t]
t.shape, torch.cat(t_list,dim=0).shape




















tensor_handler.shape


tensor_handler.metadata








sample_cols = ['BEER_SHEVA_PM10_0', 'BEER_SHEVA_PM10_values_count_0', 'BEER_SHEVA_PM10_m168', 'BEER_SHEVA_PM10_m144', 'BEER_SHEVA_PM10_m120', 'BEER_SHEVA_PM10_m96', 'BEER_SHEVA_PM10_m72', 'BEER_SHEVA_PM10_m48', 'BEER_SHEVA_PM10_m24', 'BEER_SHEVA_PM10_24', 'BEER_SHEVA_PM10_48', 'BEER_SHEVA_PM10_72', 'BEER_SHEVA_PM10_96', 'BEER_SHEVA_PM10_120', 'BEER_SHEVA_PM10_144', 'BEER_SHEVA_PM10_168', 'BEER_SHEVA_PM10_delta_0', 'BEER_SHEVA_PM10_delta_m168', 'BEER_SHEVA_PM10_delta_m144', 'BEER_SHEVA_PM10_delta_m120', 'BEER_SHEVA_PM10_delta_m96', 'BEER_SHEVA_PM10_delta_m72', 'BEER_SHEVA_PM10_delta_m48', 'BEER_SHEVA_PM10_delta_m24', 'BEER_SHEVA_PM10_delta_24', 'BEER_SHEVA_PM10_delta_48', 'BEER_SHEVA_PM10_delta_72', 'BEER_SHEVA_PM10_delta_96', 'BEER_SHEVA_PM10_delta_120', 'BEER_SHEVA_PM10_delta_144', 'BEER_SHEVA_PM10_delta_168', 'BEER_SHEVA_PM10_values_count_m168', 'BEER_SHEVA_PM10_values_count_m144', 'BEER_SHEVA_PM10_values_count_m120', 'BEER_SHEVA_PM10_values_count_m96', 'BEER_SHEVA_PM10_values_count_m72', 'BEER_SHEVA_PM10_values_count_m48', 'BEER_SHEVA_PM10_values_count_m24', 'BEER_SHEVA_PM10_values_count_24', 'BEER_SHEVA_PM10_values_count_48', 'BEER_SHEVA_PM10_values_count_72', 'BEER_SHEVA_PM10_values_count_96', 'BEER_SHEVA_PM10_values_count_120', 'BEER_SHEVA_PM10_values_count_144', 'BEER_SHEVA_PM10_values_count_168']


sample_cols[0]


data_sample_cols[data_sample_cols.index("_")+1:][data_sample_cols.index("_")+1:].index("_")+1








timestamps = debug_dataframe.index
timestamps


timestamps_new = timestamps[:5]+[]
debug_dataframe.loc[timestamps_new]





pd.to_datetime([pd.to_datetime('2006-11-05 13:00:00+0000'),pd.to_datetime('2006-11-05 14:00:00+0000')])


timestamps_new_2 = pd.to_datetime([t for t in timestamps_new]+[pd.to_datetime('2006-11-05 13:00:00+0000')])


debug_dataframe.loc[timestamps_new_2]


timestamps_new_2
debug_dataframe
existing_new_timestamps = []
for t in timestamps_new_2:
    try:
        debug_dataframe.loc[t]
        existing_new_timestamps.append(t)
    except:
        continue
timestamps_new_3 = pd.to_datetime(existing_new_timestamps)
debug_dataframe_new = debug_dataframe.loc[timestamps_new_3]


timestamps_new_2, timestamps_new_3




