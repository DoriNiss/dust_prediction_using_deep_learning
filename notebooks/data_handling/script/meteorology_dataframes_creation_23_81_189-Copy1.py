#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from MeteorologyToPandasHandler import *
import numpy as np


# import torch
# x=torch.load("../../data/datasets_23_81_169_keep_na/dataset_23_81_169_keep_na_2015_input.pkl")
# x.shape
# import matplotlib.pyplot as plt
# for c in range(23):
#     print(c)
#     plt.imshow(x[0,c])
#     plt.show()

## bad channels - 4,8 (U1000, V1000) - replacing with CAMS' U,V @ 10m


# from netCDF4 import Dataset
# file = Dataset("/work/meteogroup/cams/2003/12/D20031214_12")
# # plt.imshow(file["v10"][0])
# file.variables


data_dir = "../../data/meteorology_dataframes_23_81_189"
debug_dir = data_dir+"/debug"
debug_base_filename = "meteorology_dataframe_23_81_189_debug"
base_filename = "meteorology_dataframe_23_81_189"


meteo_handler = MeteorologyToPandasHandler(debug=False, keep_na=True, upsample_to=[81,189])
meteo_handler.params


# meteo_handler.print_param_info("SLP")


times = np.array([0])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,461,1) # -44 to 50
print("SLP:")
meteo_handler.set_idxs("SLP",[times,lats,lons])


# meteo_handler.print_param_info("Z")


times = np.array([0])
levs = np.array([6,15,20])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,461,1) # -44 to 50
print("Z:")
meteo_handler.set_idxs("Z",[times,levs,lats,lons])


# meteo_handler.print_param_info("U")
# meteo_handler.print_param_info("V")


times = np.array([0])
levs = np.array([6,15,20])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,461,1) # -44 to 50
print("U:")
meteo_handler.set_idxs("U",[times,levs,lats,lons])
print("\nV:")
meteo_handler.set_idxs("V",[times,levs,lats,lons])


# meteo_handler.print_param_info("PV")


times = np.array([0])
levs = np.array([2,3,4,5,6,7,8])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,461,1) # -44 to 50
print("PV:")
meteo_handler.set_idxs("PV",[times,levs,lats,lons])


# meteo_handler.print_param_info("aod550")


time = np.array([0])
latitude = np.arange(70,29,-1) # 20 to 60
longitude = np.arange(136,231,1) # -44 to 50
print("aod550:")
meteo_handler.set_idxs("aod550",[time,latitude,longitude])
print("\nduaod550:")
meteo_handler.set_idxs("duaod550",[time,latitude,longitude])
print("\naermssdul:")
meteo_handler.set_idxs("aermssdul",[time,latitude,longitude])
print("\naermssdum:")
meteo_handler.set_idxs("aermssdum",[time,latitude,longitude])
print("\nu10:")
meteo_handler.set_idxs("u10",[time,latitude,longitude])
print("\nv10:")
meteo_handler.set_idxs("v10",[time,latitude,longitude])


meteo_handler.load_and_save_yearly_data(data_dir, base_filename, njobs=3)








# from scipy import interpolate
# import scipy.ndimage
# from PIL import Image
# import numpy as np

# upsample_to = [81,189]

# original = np.zeros([1,41,95])
# print(original.shape)
# upsampled = scipy.ndimage.zoom(original[0], 2, order=3)
# print(upsampled.shape)
# upsampled = Image.fromarray(upsampled).resize([upsample_to[-1],upsample_to[-2]])
# upsampled = np.array(upsampled)
# print(upsampled.shape)




