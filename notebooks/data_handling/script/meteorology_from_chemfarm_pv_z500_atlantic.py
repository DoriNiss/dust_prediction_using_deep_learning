#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from Meteorology_to_pandas_handler import *
import numpy as np


# meteo_handler = Meteorology_to_pandas_handler(debug=True)
params = ["PV", "Z"]
meteo_handler = Meteorology_to_pandas_handler(params=params, debug=False)
meteo_handler.params


meteo_handler.print_param_info("PV")


times = np.array([0])
levs = np.array([2,3,4,5,6,7,8])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,441,1) # -44 to 40
meteo_handler.set_idxs("PV",[times,levs,lats,lons])


meteo_handler.print_param_info("Z")


times = np.array([0])
levs = np.array([15]) # only 500
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,441,1) # -44 to 40
meteo_handler.set_idxs("Z",[times,levs,lats,lons])


meteo_dataframe = meteo_handler.load_data()


meteo_dataframe["PV"][0].shape





# loaded = torch.load(path_to_dir+"/meteorology_pv_z500_dataframe_2000.pkl")
# loaded["PV"][0].shape, loaded["Z"][10].shape





print("Created dataframe of size ",len(meteo_dataframe), "with the following dates:")
print(meteo_dataframe[:5].index)
print(meteo_dataframe[-5:].index)
print("PV shape: ",meteo_dataframe[0:5]["PV"][2].shape, ", Z shape: ",meteo_dataframe[0:5]["Z"][2].shape)


path_to_dir = "../../data/pv_to_z500_wide"
base_filename = "meteorology_pv_z500_dataframe"
meteo_handler.save_dataframe_by_years(meteo_dataframe, path_to_dir, base_filename)





# check sizes


import torch
path_to_dir = "../../data/pv_to_z500_wide"

for y in range(2000,2022):
    loaded = torch.load(path_to_dir+"/meteorology_pv_z500_dataframe_"+str(y)+".pkl")
    print(y,":",len(loaded))


print(2928+2920+2920+2920+2928+2920+2920+2920+2928+2920+2920+2920+2928+2920+2920+2920+2928+2920+2920+2598)




