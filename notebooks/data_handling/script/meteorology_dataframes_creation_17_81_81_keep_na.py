#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from MeteorologyToPandasHandler import *
import numpy as np


meteo_handler = MeteorologyToPandasHandler(debug=False, keep_na=True)
meteo_handler.params


# meteo_handler.print_param_info("SLP")


times = np.array([0])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(360,441,1) # 0 to 40
meteo_handler.set_idxs("SLP",[times,lats,lons])


# meteo_handler.print_param_info("Z")


times = np.array([0])
levs = np.array([6,15,20])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(360,441,1) # 0 to 40
meteo_handler.set_idxs("Z",[times,levs,lats,lons])


# meteo_handler.print_param_info("U")


times = np.array([0])
levs = np.array([6,15,20])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(360,441,1) # 0 to 40
meteo_handler.set_idxs("U",[times,levs,lats,lons])


# meteo_handler.print_param_info("V")


times = np.array([0])
levs = np.array([6,15,20])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(360,441,1) # 0 to 40
meteo_handler.set_idxs("V",[times,levs,lats,lons])


# meteo_handler.print_param_info("PV")


times = np.array([0])
levs = np.array([2,3,4,5,6,7,8])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(360,441,1) # 0 to 40
meteo_handler.set_idxs("PV",[times,levs,lats,lons])


meteo_dataframe = meteo_handler.load_data()


print("Length of dataframe:", len(meteo_dataframe))
print("Sample SLP shape:",meteo_dataframe["SLP"][0].shape)
print("Sample Z shape:",meteo_dataframe["Z"][0].shape)
print("Sample U shape:",meteo_dataframe["U"][0].shape)
print("Sample V shape:",meteo_dataframe["V"][0].shape)
print("Sample PV shape:",meteo_dataframe["PV"][0].shape)
print("Dates:")
print(meteo_dataframe[:5].index)
print(meteo_dataframe[-5:].index)


path_to_dir = "../../data/meteorology_dataframes_17_81_81_keep_na"
base_filename = "meteorology_df_17_81_81_keep_na"
meteo_handler.save_dataframe_by_years(meteo_dataframe, path_to_dir, base_filename)




