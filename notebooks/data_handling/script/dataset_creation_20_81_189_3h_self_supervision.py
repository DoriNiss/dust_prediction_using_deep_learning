#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DatasetHandler import *
from utils.files_loading import *
from data_handlers.tensors_labeling_and_averaging import *
from tqdm import tqdm


# years = [y for y in range(2003, 2019)]
# dataframes_dir = "../../data/meteorology_dataframes_20_81_189_3h/"
# dataframes_dir_debug = "../../data/meteorology_dataframes_20_81_189_3h/debug/"


# dataframes_descriptions_paths = [
#     dataframes_dir+"metadata/meteorology_dataframes_20_81_189_3h_description.pkl",
# ]

# dataframes_descriptions = [torch.load(path) for path in dataframes_descriptions_paths]
# dataframes_descriptions[0]["target"] = dataframes_descriptions[0]["input"]


# datasets_dir = "../../data/datasets_20_81_189_3h_self_supervision"
# datasets_metadata_path = datasets_dir+"/metadata/datasets_20_81_189_3h_self_supervision_metadata.pkl"
# torch.save(dataframes_descriptions[0],datasets_metadata_path)
# dataframes_descriptions = [torch.load(datasets_metadata_path)]
# dataframes_descriptions





debug_mode = False

years = [y for y in range(2003, 2019)] if not debug_mode else [y for y in range(2002, 2004)]
dataframes_dir = "../../data/meteorology_dataframes_20_81_189_3h/"
dataframes_dir_debug = "../../data/meteorology_dataframes_20_81_189_3h/debug/"
datasets_dir = "../../data/datasets_20_81_189_3h_self_supervision"
datasets_dir_debug = "../../data/datasets_20_81_189_3h_self_supervision/debug"
datasets_metadata_path = datasets_dir+"/metadata/datasets_20_81_189_3h_self_supervision_metadata.pkl"


dataframes_descriptions_paths = [
    datasets_metadata_path,
]
dataframes_descriptions = [torch.load(path) for path in dataframes_descriptions_paths]

# dust_data_path = "../../data/dust_20000101to20213006_3h_7days_future.pkl"


dataframes_paths = []
for y in years:
    if debug_mode:
        dataframes_paths.append([
            dataframes_dir_debug+"meteorology_dataframe_20_81_189_3h_debug_"+str(y)+".pkl",
#             dust_data_path
        ])
    else:
        dataframes_paths.append([
            dataframes_dir+"meteorology_dataframe_20_81_189_3h_"+str(y)+".pkl",
#             dust_data_path
        ])

        
datasets_arguments = []
for y in years:        
    datasets_arguments.append(
        {
            "cols_input": ["SLP","Z","U","V","PV","aod550","duaod550","aermssdul","aermssdum","u10","v10"],
#             "cols_target": ["SLP"],
            "cols_target": [],
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
        
base_filename = "dataset_20_81_189_3h_self_supervision"
base_filename_debug = "datasets_20_81_189_3h_self_supervision_debug"
save_as_list = []
for i in range(len(dataframes_paths)):
    if debug_mode:
        save_as_list.append({
            "dir_path": datasets_dir_debug,
            "base_filename": f"{base_filename_debug}_{years[i]}"
        })
    else:
        save_as_list.append({
            "dir_path": datasets_dir,
            "base_filename": f"{base_filename}_{years[i]}"
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


2919*16+8*4


sample_year = 2015

sample_input = torch.load(f"{datasets_dir}/{base_filename}_{sample_year}_input.pkl")
sample_timestamps = torch.load(f"{datasets_dir}/{base_filename}_{sample_year}_timestamps.pkl")

print(sample_input.shape,len(sample_timestamps))


sample_input[5],sample_input[6]





def load_stacked_inputs_targets_timestamps_from_years_list(years_list, path_dir, base_filename, ignore_targets=False):
    targets_paths = get_paths_from_years_list(years_list, path_dir, base_filename, "target")
    inputs_paths = get_paths_from_years_list(years_list, path_dir, base_filename, "input")
    timestamps_paths = get_paths_from_years_list(years_list, path_dir, base_filename, "timestamps")
    targets = None
    if not ignore_targets: 
        print("Creating one tensor from all targets...")
        targets = get_stacked_tensor_from_paths(targets_paths)
    print("Creating one tensor from all inputs...")
    inputs = get_stacked_tensor_from_paths(inputs_paths)
    timestamps = get_stacked_timestamps_from_paths(timestamps_paths)
    if not ignore_targets: 
        print(f"\n\nDone! Result sizes: inputs: {inputs.shape}, targets: {targets.shape}, timestamps: {len(timestamps)}")
    else:
        print(f"\n\nDone! Result sizes: inputs: {inputs.shape}, timestamps: {len(timestamps)}")
    return inputs, targets, timestamps
        


inputs_all,_,timestamps_all = load_stacked_inputs_targets_timestamps_from_years_list(years,
                                                                                     datasets_dir, 
                                                                                     base_filename,
                                                                                     ignore_targets=True)
inputs_all.shape,len(timestamps_all)


torch.save(inputs_all,f"{datasets_dir}/{base_filename}_all_inputs.pkl")
torch.save(timestamps_all,f"{datasets_dir}/{base_filename}_all_timestamps.pkl")




