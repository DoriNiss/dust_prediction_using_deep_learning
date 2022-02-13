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
meteorology_dataframe_dir = f"{data_dir}/meteorology_dataframes_62_81_189"
result_dir_debug = f"{data_dir}/meteorology_tensors_62_81_189/debug"
base_filename_debug = f"{result_dir_debug}/meteorology_tensor_62_81_189"
result_dir = f"{data_dir}/meteorology_tensors_62_81_189"
base_filename = f"{result_dir}/meteorology_tensor_62_81_189"
metadata_filename = f"{meteorology_dataframe_dir}/meteorology_df_metadata.pkl"
# timestamps_dir = f"{data_dir}/dust_55520_108_2_339"


metadata_df = torch.load(metadata_filename)
# metadata_df


metadata = {}
metadata["params"] = {p:
                      {
                          "title":metadata_df["params"][p]["title"],
                          "folder_path":metadata_df["params"][p]["folder_path"],
                          "file_prefix":metadata_df["params"][p]["file_prefix"]
                      }
for p in metadata_df["params"].keys()}
metadata['df_infill_missing_res'] = metadata_df['infill_missing_res']
metadata['df_infill_missing_params'] = metadata_df['infill_missing_params']
metadata['df_infill_missing_values'] = metadata_df['infill_missing_values']
lats_dict = {i:idx for i,idx in enumerate(np.arange(10,50.5,0.5))}
lons_dict = {i:idx for i,idx in enumerate(np.arange(-32,62.5,0.5))}
# metadata





dims_cols_strings = {1:[c_str for c_str in metadata["params"].keys()]}
lats = np.array([i for i in lats_dict.values()])
lons = np.array([i for i in lons_dict.values()])


metadata











# load 2 following tensors
# cat tensors
# interpolate missing res
# save the first, keep the last one in memory

# create yearly or one big tensor





def load_interpolate_res_and_save_batched_tensors(base_filename,years,batches,create_metadata=False):
    # Assuming all files exist
    t,timestamps,name_t,name_timestampe = get_dataset_from_dataframe(base_filename,years[0],batches[0],
                                                                     create_metadata=create_metadata)
    for year in years:
        for batch in batches:
            next_year,next_batch=calculate_next_year_and_batch(years,batches,year,batch)         
            next_t,next_timestamps,next_name_t,next_name_timestampe = get_dataset_from_dataframe(
                base_filename,next_year,next_batch,create_metadata=create_metadata)
            t = interpolate_end_of_t_where_value(t,next_t)
            torch.save(t,name_t)
            torch.save(timestamps,name_timestampe)
            name_str_print = name_t[-21:-11][name_t[-21:-11].index("_"):]
            print(f"... Saved '...{name_str_print}...', {t.shape}")
            t,timestamps,name_t,name_timestampe = next_t,next_timestamps,next_name_t,next_name_timestampe
    print("... Done!")
            
def calculate_next_year_and_batch(years,batches,current_year,current_batch):
    if current_batch>=batches[-1]:
        if current_year>=years[-1]:
            next_year,next_batch=years[-1],batches[-1]
        else:
            next_year,next_batch=current_year+1,0
    else:
        next_year,next_batch=current_year,current_batch+1
    return next_year,next_batch

def interpolate_end_of_t_where_value(t1,t2):
    # Assuming the last row could not have been interpoalted earlier in df creation
    invalid_map = t1[-1,-1]
    idxs = t1[-1,:]==invalid_map
    t_interpolated = 0.5*(t1[-2]+t2[0])
    t1[-1][idxs]=t_interpolated[idxs]
    return t1

def get_dataset_from_dataframe(base_filename,year,batch,create_metadata=False):
    meteorology_dataframe_filename = f"{meteorology_dataframe_dir}/meteorology_df_{year}_b{batch}.pkl"
    df = torch.load(meteorology_dataframe_filename)
    filename_tensor = f"{base_filename}_{year}_b{batch}"
    handler = DatasetHandler_DataframeToTensor_Meteorology(
        df, dims_cols_strings=dims_cols_strings, metadata=metadata, timestamps=None, 
        save_base_filename=filename_tensor,
        invalid_col_fill=-777, param_shape=[81,189], lons=lons, lats=lats, save_timestamps=True,
        old_format_channels_dict=None, verbose=0
    )
    if create_metadata: handler.create_metadata()
    return handler.get_dataset_from_timestamps(df.index, suffix=None)














base_filename


years=list(range(2000,2021))
batches = list(range(98))
Parallel(n_jobs=11,verbose=100)(delayed(load_interpolate_res_and_save_batched_tensors)(
    base_filename,[year],batches) for year in years
)
# Interpolates between batches, will need to interpolate last batch of every year!


# Build yearly tensors
def combine_year(base_filename,year,batches):
    tensors_list,timestamps_lists=[],[]
    for batch in batches:
        tensors_list.append(torch.load(f"{base_filename}_{year}_b{batch}_tensor.pkl"))
        timestamps_lists.append(torch.load(f"{base_filename}_{year}_b{batch}_timestamps.pkl"))
    yearly_tensor, yearly_timestamps = DatasetHandler_DataframeToTensor_Meteorology.merge_by_timestamps(
        tensors_list, timestamps_lists)
    torch.save(yearly_tensor,f"{base_filename}_{year}_tensor_full.pkl")
    torch.save(yearly_timestamps,f"{base_filename}_{year}_timestamps_full.pkl")
    print(f"#### Year: {year}: shape {yearly_tensor.shape}, timestamps len {len(yearly_timestamps)}")
    
years = list(range(2000,2021))
batches = list(range(98))
Parallel(n_jobs=3,verbose=100)(delayed(combine_year)(
    base_filename,year,batches) for year in years
)


# Interpolate last batch of years

years = list(range(2000,2021))
batches = list(range(98))
t_current_year = torch.load(f"{base_filename}_{years[0]}_tensor_full.pkl")
for next_year in years[1:]:
    current_year = next_year-1 # Assuming no skip years
    t_next_year = torch.load(f"{base_filename}_{next_year}_tensor_full.pkl")
    t_current_year = interpolate_end_of_t_where_value(t_current_year,t_next_year)
    torch.save(t_current_year,f"{base_filename}_{current_year}_tensor_full.pkl")
    print(f"#### Year {current_year} interpolated and saved!") 
    t_current_year=t_next_year
    


# Build metadata
metadata_new = torch.load("../../data/meteorology_tensors_62_81_189/debug/meteorology_tensor_62_81_189_hanlder_2000_b0_metadata.pkl")
channels_dict = {c: dims_cols_strings[1][c] for c in range(len(metadata_new["params"].keys()))}
metadata_new["dims"]["channels"] = channels_dict
metadata["dims"] = metadata_new["dims"]
torch.save(metadata,f"{base_filename}_metadata.pkl")


metadata


# Build tensor per channel
channels = metadata["dims"]["channels"].items()
save_to_dir = f"{data_dir}/meteorology_tensors_62_81_189/per_channel"
save_as_base_filename = f"{save_to_dir}/meteorology_tensors_1_81_189"
years = list(range(2000,2021))
to_save_timestamps = True
for c,channel in channels:
    print(f"#### {c,channel}...")
    tensors_list,timestamps_lists=[],[]
    for year in tqdm(years):
        tensors_list.append(torch.load(f"{base_filename}_{year}_tensor_full.pkl")[:,c,:,:].unsqueeze(1))
        timestamps_lists.append(torch.load(f"{base_filename}_{year}_timestamps_full.pkl"))
        full_tensor, full_timestamps = DatasetHandler_DataframeToTensor_Meteorology.merge_by_timestamps(
            tensors_list, timestamps_lists)
        tensors_list,timestamps_lists = [full_tensor],[full_timestamps]
    if to_save_timestamps: 
        torch.save(full_timestamps,f"{save_as_base_filename}_general_timestamps.pkl")
        to_save_timestamps = False
    torch.save(full_tensor,f"{save_as_base_filename}_{channel}.pkl")
    print(f"......Done! saved {channel}, shape {full_tensor.shape}, timestamps length {len(full_timestamps)}")


metadata["dims"]["general"] = f"[timestamps,channels,lons,lats] = [{len(full_timestamps)}, 62, 81, 189]"
metadata


torch.save(metadata,f"{base_filename}_metadata.pkl")














#### OLD


# # Adding last batch fix
# years=list(range(2000,2021))
# batches = [0,96,97]
# load_interpolate_res_and_save_batched_tensors(base_filename,years,batches)


# Creating metadata
years=[2000]
batches = [0]
load_interpolate_res_and_save_batched_tensors(f"{base_filename}_hanlder",years,batches,create_metadata=True)


metadata_new = torch.load("../../data/meteorology_tensors_62_81_189/debug/meteorology_tensor_62_81_189_hanlder_2000_b0_metadata.pkl")
metadata_new


len(metadata_new["params"].keys())


channels_dict = {c: dims_cols_strings[1][c] for c in range(len(metadata_new["params"].keys()))}


metadata_new["dims"]["channels"] = channels_dict
metadata_new["dims"]


torch.save(metadata_new,f"{base_filename}_metadata.pkl")


torch.load(f"{base_filename}_metadata.pkl")





# data_dir = "../../data"
# meteorology_dataframe_dir = f"{data_dir}/meteorology_dataframes_20_81_189_3h"
# result_dir = f"{data_dir}/dataset_input_20_81_189_old_dust_108_2_339"
# file_basename = f"{result_dir}/meteorology_20_81_189_old"
# metadata_filename = f"{data_dir}/meteorology_dataframes_20_81_189_3h/metadata/meteorology_dataframes_20_81_189_3h_description.pkl"
# timestamps_dir = f"{data_dir}/dust_55520_108_2_339"


# moved to another dir (not "debug"...)
tensors_list,timestamps_lists=[],[]
years = list(range(2000,2021))
batches = list(range(98))
print("Loading tensors...")
for year in years:
    print(f"#### Year: {year}")
    for batch in tqdm(batches):
        tensors_list.append(torch.load(f"{base_filename}_{year}_b{batch}_tensor.pkl"))
        timestamps_lists.append(torch.load(f"{base_filename}_{year}_b{batch}_timestamps.pkl"))
    print("   Combining tensors...")
    yearly_tensor, yearly_timestamps = DatasetHandler_DataframeToTensor_Meteorology.merge_by_timestamps(
        tensors_list, timestamps_lists)
    print(yearly_tensor.shape, len(yearly_timestamps))
    tensors_list,timestamps_lists=[],[]
    torch.save(yearly_tensor,f"{base_filename}_{year}_tensor_full.pkl")
    torch.save(yearly_timestamps,f"{base_filename}_{year}_timestamps_full.pkl")


# died in year 2015...
saveas = f"{data_dir}/meteorology_tensors_62_81_189/meteorology_tensor_62_81_189"
tensors_list,timestamps_lists=[],[]
years = list(range(2000,2021))
print("Loading tensors...")
for year in years:
    print(f"#### Year: {year}")
    tensors_list.append(torch.load(f"{base_filename}_{year}_tensor_full.pkl"))
    timestamps_lists.append(torch.load(f"{base_filename}_{year}_timestamps_full.pkl"))
print("   Combining tensors...")
full_tensor, full_timestamps = DatasetHandler_DataframeToTensor_Meteorology.merge_by_timestamps(
    tensors_list, timestamps_lists)
print(full_tensor.shape, len(full_timestamps))
torch.save(full_tensor,f"{saveas}_tensor_full.pkl")
torch.save(yearly_timestamps,f"{saveas}_timestamps_full.pkl")


# Per channel:


base_filename = "../../data/meteorology_tensors_62_81_189/meteorology_tensor_62_81_189"
metadata_new = torch.load(f"{base_filename}_metadata.pkl")
channels = metadata_new["dims"]["channels"].items()
channels


# Save all years per channel
save_to_dir = f"{data_dir}/meteorology_tensors_62_81_189/per_channel"
save_as_base_filename = f"{save_to_dir}/meteorology_tensors_1_81_189"
years = list(range(2000,2021))
to_save_timestamps = True
for c,channel in channels:
    print(f"#### {c,channel}...")
    tensors_list,timestamps_lists=[],[]
    for year in tqdm(years):
        tensors_list.append(torch.load(f"{base_filename}_{year}_tensor_full.pkl")[:,c,:,:].unsqueeze(1))
        timestamps_lists.append(torch.load(f"{base_filename}_{year}_timestamps_full.pkl"))
        full_tensor, full_timestamps = DatasetHandler_DataframeToTensor_Meteorology.merge_by_timestamps(
            tensors_list, timestamps_lists)
        tensors_list,timestamps_lists = [full_tensor],[full_timestamps]
    if to_save_timestamps: 
        torch.save(full_timestamps,f"{save_as_base_filename}_general_timestamps.pkl")
        to_save_timestamps = False
    torch.save(full_tensor,f"{save_as_base_filename}_{channel}.pkl")
    print(f"......Done! saved {channel}, shape {full_tensor.shape}, timestamps length {len(full_timestamps)}")






















