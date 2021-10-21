#!/usr/bin/env python
# coding: utf-8

import torch
import sys
sys.path.insert(0, '../../packages/data_handlers')
from DatasetHandler import *
import numpy as np


years = [y for y in range(2000, 2022)]
dataframes_dir = "../../data/meteorology_dataframes_23_81_169_keep_na/"


dataframes_descriptions_paths = [
    dataframes_dir+"metadata/meteorology_dataframes_23_81_169_keep_na_description.pkl",
    "../../data/dust_description_pm10_BeerSheva_20000101_20210630_6h.pkl"
]
dataframes_descriptions = [torch.load(path) for path in dataframes_descriptions_paths]


dust_data_path = "../../data/dust_20000101to20213006_6h_keep_na.pkl"


dataframes_paths = []
for y in years:
    dataframes_paths.append([
        dataframes_dir+"meteorology_dataframe_23_81_169_keep_na_"+str(y)+".pkl",
        dust_data_path
    ])

        
datasets_arguments = []
for y in years:        
    datasets_arguments.append(
        {
            "cols_input": ["SLP","Z","U","V","PV","aod550","duaod550","aermssdul","aermssdum"],
            "cols_target": ["dust_0","delta_0"],
            "dataframes_descriptions": dataframes_descriptions,
            "keep_na": True,
            "include_all_timestamps_between": True,
            "all_timestamps_intervals": "6h",
            "cols_channels_input": None,
            "cols_channels_target": None,
            "as_float32": True,
            "wanted_year": y
        })

        
datasets_dir = "../../data/datasets_23_81_169_keep_na"
save_as_list = []
for i in range(len(dataframes_paths)):
    save_as_list.append({
        "dir_path": datasets_dir,
        "base_filename": "dataset_23_81_169_keep_na_"+str(years[i])
    })


DatasetHandler.create_and_save_datasets_from_paths(dataframes_paths, datasets_arguments, save_as_list, njobs=3)




