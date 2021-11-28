#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DatasetHandler import *
from utils.files_loading import *
from data_handlers.tensors_labeling_and_averaging import *
from tqdm import tqdm


years = [y for y in range(2003, 2019)]
dataframes_dir = "../../data/meteorology_dataframes_20_81_189_3h/"
dataframes_dir_debug = "../../data/meteorology_dataframes_20_81_189_3h/debug/"


dataframes_descriptions_paths = [
    "../../data/datasets_20_81_189_3h_7days_future_with_history/metadata/" \
    "dataset_20_81_189_3h_7days_future_with_history_2004_descriptions.pkl"
]
dataframes_descriptions = [torch.load(path) for path in dataframes_descriptions_paths]


dust_data_path = "../../data/dust_20000101to20213006_3h_7days_future_with_history.pkl"


dataframes_paths = []
for y in years:
    dataframes_paths.append([
        dataframes_dir+"meteorology_dataframe_20_81_189_3h_"+str(y)+".pkl",
#         dataframes_dir_debug+"meteorology_dataframe_20_81_189_3h_debug_"+str(y)+".pkl",
        dust_data_path
    ])

        
datasets_arguments = []
for y in years:        
    datasets_arguments.append(
        {
            "cols_input": ["SLP","Z","U","V","PV","aod550","duaod550","aermssdul","aermssdum","u10","v10"],
            "cols_target": [dataframes_descriptions[0]["target"][i]["df_col"]  for i in range(36)],
            "dataframes_descriptions": dataframes_descriptions,
            "keep_na": False,
            "replace_na_in_target": None,
            "include_all_timestamps_between": None,
            "all_timestamps_intervals": "3h",
            "cols_channels_input": None,
            "cols_channels_target": None,
            "as_float32": True,
            "wanted_year": y
        })
        
datasets_dir = "../../data/datasets_20_81_189_3h_7days_future_with_history"
datasets_dir_debug = "../../data/datasets_20_81_189_3h_7days_future_with_history/debug"
base_filename = "dataset_20_81_189_3h_7days_future_with_history"
base_filename_debug = "dataset_20_81_189_3h_7days_future_with_history_debug"
save_as_list = []
for i in range(len(dataframes_paths)):
    save_as_list.append({
        "dir_path": datasets_dir,
#         "dir_path": datasets_dir_debug,
        "base_filename": f"{base_filename}_{years[i]}"
#         "base_filename": f"{base_filename_debug}_{years[i]}"
    })





DatasetHandler.create_and_save_datasets_from_paths(dataframes_paths, datasets_arguments, save_as_list, njobs=2)


# dataframes_paths = []
# for y in years:
#     dataframes_paths.append([
#         dataframes_dir_debug+"meteorology_dataframe_20_81_189_3h_debug_"+str(y)+".pkl",
#         dust_data_path
#     ])
# print(dataframes_paths[0])
# DatasetHandler.create_and_save_one_dataset_from_path(dataframes_paths[0], datasets_arguments[0], save_as_list[0])





sample_year = 2015

sample_input = torch.load(f"{datasets_dir}/{base_filename}_{sample_year}_input.pkl")
sample_target = torch.load(f"{datasets_dir}/{base_filename}_{sample_year}_target.pkl")
sample_timestamps = torch.load(f"{datasets_dir}/{base_filename}_{sample_year}_timestamps.pkl")

print(sample_input.shape,sample_target.shape,len(sample_timestamps))


sample_target[5],sample_target[6]


sample_input[5],sample_input[6]








inputs_all,targets_all,timestamps_all = load_stacked_inputs_targets_timestamps_from_years_list(years,
                                                                                               datasets_dir, 
                                                                                               base_filename)
inputs_all.shape,targets_all.shape,len(timestamps_all)


torch.save(inputs_all,f"{datasets_dir}/{base_filename}_all_inputs.pkl")
torch.save(targets_all,f"{datasets_dir}/{base_filename}_all_targets.pkl")
torch.save(timestamps_all,f"{datasets_dir}/{base_filename}_all_timestamps.pkl")


targets_all[targets_all[:,0]>=74.3].shape


30872/16


timestamps_all[-1]




