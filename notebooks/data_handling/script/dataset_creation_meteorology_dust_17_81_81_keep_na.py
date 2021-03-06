#!/usr/bin/env python
# coding: utf-8

import torch
import sys
sys.path.insert(0, '../../packages/data_handlers')
from DatasetHandler import *
import numpy as np


# torch.save(dataframes[0][2900:3100],"../../data/meteorology_dataframes_17_81_81_keep_na/meteorology_df_17_81_81_keep_na_debug.pkl")


dataframes_paths_meteo = ["../../data/meteorology_dataframes_17_81_81_keep_na/meteorology_df_17_81_81_keep_na_2000.pkl",
                    "../../data/meteorology_dataframes_17_81_81_keep_na/meteorology_df_17_81_81_keep_na_2001.pkl"]
dataframes_path_meteo_debug = "../../data/meteorology_dataframes_17_81_81_keep_na/meteorology_df_17_81_81_keep_na_debug.pkl"
dataframe_path_dust = "../../data/dust_20000101to20213006_0_keep_na.pkl"

dataframe0 = torch.load(dataframes_path_meteo_debug)
dataframe1 = torch.load(dataframe_path_dust)

dataframes = [dataframe0, dataframe1]


dataframes[0].index[0], dataframes[1].index[0]


dataset_handler = DatasetHandler(dataframes,["SLP","Z","U","V","PV"],["PV","U","V"],["Z"],"bla",
                                 cols_channels_input=None, cols_channels_target=None)
                


dataset_handler.cols_target, dataset_handler.cols_channels_target


import pandas as pd
date1 = pd.to_datetime("2000-02-01")
date2 = pd.to_datetime("2000-02-02")
date1, date2, date1<date2


series = pd.date_range(start=date1, end=date2, freq="3h", tz="UTC")
series




