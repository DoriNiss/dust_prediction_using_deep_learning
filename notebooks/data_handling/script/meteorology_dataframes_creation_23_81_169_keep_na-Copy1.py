#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from MeteorologyToPandasHandler import *
import numpy as np


data_dir = "../../data/meteorology_dataframes_23_81_169_keep_na"
debug_dir = data_dir+"/debug"
debug_base_filename = "meteorology_dataframe_23_81_169_keep_na_debug"
base_filename = "meteorology_dataframe_23_81_169_keep_na"


meteo_handler = MeteorologyToPandasHandler(debug=False, keep_na=True)
meteo_handler.params


# meteo_handler.print_param_info("SLP")


times = np.array([0])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,441,1) # -44 to 40
print("SLP:")
meteo_handler.set_idxs("SLP",[times,lats,lons])


# meteo_handler.print_param_info("Z")


times = np.array([0])
levs = np.array([6,15,20])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,441,1) # -44 to 40
print("Z:")
meteo_handler.set_idxs("Z",[times,levs,lats,lons])


# meteo_handler.print_param_info("U")
# meteo_handler.print_param_info("V")


times = np.array([0])
levs = np.array([0,6,15,20])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,441,1) # -44 to 40
print("U:")
meteo_handler.set_idxs("U",[times,levs,lats,lons])
print("\nV:")
meteo_handler.set_idxs("V",[times,levs,lats,lons])


# meteo_handler.print_param_info("PV")


times = np.array([0])
levs = np.array([2,3,4,5,6,7,8])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,441,1) # -44 to 40
print("PV:")
meteo_handler.set_idxs("PV",[times,levs,lats,lons])


# meteo_handler.print_param_info("aod550")


time = np.array([0])
latitude = np.arange(70,29,-1) # 20 to 60
longitude = np.arange(136,221,1) # -44 to 40
print("aod550:")
meteo_handler.set_idxs("aod550",[time,latitude,longitude])
print("\nduaod550:")
meteo_handler.set_idxs("duaod550",[time,latitude,longitude])
print("\naermssdul:")
meteo_handler.set_idxs("aermssdul",[time,latitude,longitude])
print("\naermssdum:")
meteo_handler.set_idxs("aermssdum",[time,latitude,longitude])


meteo_handler.load_and_save_yearly_data(data_dir, base_filename, njobs=3)













