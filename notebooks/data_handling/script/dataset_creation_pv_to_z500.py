#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from Meteorology_to_pandas_handler import *
import numpy as np


filename = "../../data/pv_to_z500_wide/meteorology_pv_z500_dataframe_all.pkl"
print("Created dataframe of size ",len(meteo_dataframe), "with the following dates:")
print(meteo_dataframe[:5].index)
print(meteo_dataframe[-5:].index)
print("PV shape: ",meteo_dataframe[0:5]["PV"][2].shape, ", Z shape: ",meteo_dataframe[0:5]["Z"][2].shape)
print("Saved to ",filename)


def create_pv_to_z500_dataset_by_years(dataframe_all, years): # to be added to DatasetHandler
    data_wanted_years = 
    return pv_tensor, 







