#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from Meteorology_to_pandas_handler import *
import numpy as np


# meteo_handler = Meteorology_to_pandas_handler(debug=True)
meteo_handler = Meteorology_to_pandas_handler()
meteo_handler.params


meteo_handler.print_param_info("SLP")


times = np.array([0])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(360,441,1) # 0 to 40
meteo_handler.set_idxs("SLP",[times,lats,lons])


meteo_handler.print_param_info("Z")


times = np.array([0])
levs = np.array([6,15,20])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(360,441,1) # 0 to 40
meteo_handler.set_idxs("Z",[times,levs,lats,lons])


meteo_handler.print_param_info("U")


times = np.array([0])
levs = np.array([6,15,20])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(360,441,1) # 0 to 40
meteo_handler.set_idxs("U",[times,levs,lats,lons])


meteo_handler.print_param_info("V")


times = np.array([0])
levs = np.array([6,15,20])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(360,441,1) # 0 to 40
meteo_handler.set_idxs("V",[times,levs,lats,lons])


meteo_handler.print_param_info("PV")


times = np.array([0])
levs = np.array([2,3,4,5,6,7,8])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(360,441,1) # 0 to 40
meteo_handler.set_idxs("PV",[times,levs,lats,lons])


meteo_dataframe = meteo_handler.load_data()


# filename = "../../data/meteorology_dataframe_debug_20000101to20210630.pkl"
filename = "../../data/meteorology_dataframe_20000101to20210630.pkl"
meteo_handler.save_dataframe(meteo_dataframe,filename)


# meteo_dataframe.describe()


meteo_dataframe[0:5]["PV"][2].shape




