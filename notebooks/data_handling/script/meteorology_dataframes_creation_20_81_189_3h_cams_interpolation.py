#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from MeteorologyToPandasHandler import *
import numpy as np
import pandas as pd


data_dir = "../../data/meteorology_dataframes_20_81_189_3h"
base_filename = "meteorology_dataframe_20_81_189_3h"
debug_dir = data_dir+"/debug"
debug_base_filename = "meteorology_dataframe_20_81_189_3h_debug"


# connection closed before finishing these 2 years
dates = pd.date_range(start="2014-01-01 00:00", 
                      end="2015-12-31 21:00", 
                      tz="UTC", freq="3h")   
dates[:3],dates[-3:]


meteo_handler = MeteorologyToPandasHandler(debug=False, keep_na=False, add_cams=True, upsample_to=[81,189],
                interpolate_to_3h=True,dates=dates)
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
levs = np.array([5,6,7,8])
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


meteo_handler.load_and_save_yearly_data(data_dir, base_filename, njobs=2)





# meteo_handler.load_and_save_yearly_data(debug_dir, debug_base_filename, njobs=3)
# meteo_handler.load_and_save_yearly_data(data_dir, base_filename, njobs=2)





def validate_year(year):
    filename = f"../../data/meteorology_dataframes_20_81_189_3h/meteorology_dataframe_20_81_189_3h_{year}.pkl"
    df_sample = torch.load(filename)
    print(f"\n#### Sample year: {year}")
    print(f"Length: {len(df_sample)}")
    print(f"Length: {len(df_sample)}")
    print(f"Some shapes: Z:{df_sample['Z'][0].shape},PV:{df_sample['PV'][0].shape},aermssdum:{df_sample['aermssdum'][0].shape}")
    print("Some rows:")
    print(df_sample[:3])
    print(df_sample[-3:])
    return df_sample


validate_year(2013)











meteo_handler = MeteorologyToPandasHandler(debug=True, keep_na=False, add_cams=True, upsample_to=[81,189],
                interpolate_to_3h=True,dates=None)

times = np.array([0])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,461,1) # -44 to 50
print("SLP:")
meteo_handler.set_idxs("SLP",[times,lats,lons])

times = np.array([0])
levs = np.array([6,15,20])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,461,1) # -44 to 50
print("Z:")
meteo_handler.set_idxs("Z",[times,levs,lats,lons])

times = np.array([0])
levs = np.array([6,15,20])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,461,1) # -44 to 50
print("U:")
meteo_handler.set_idxs("U",[times,levs,lats,lons])
print("\nV:")
meteo_handler.set_idxs("V",[times,levs,lats,lons])

times = np.array([0])
levs = np.array([5,6,7,8])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(272,461,1) # -44 to 50
print("PV:")
meteo_handler.set_idxs("PV",[times,levs,lats,lons])

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

meteo_handler.load_and_save_yearly_data(debug_dir, debug_base_filename, njobs=1)





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


# df_debug = torch.load("../../data/meteorology_dataframes_20_81_189_3h/debug/meteorology_dataframe_20_81_189_3h_debug_2003.pkl")


# df_debug[-10:]


# import matplotlib.pyplot as plt


# idxs = [-11,-10,-9]
# samples = []
# for idx in idxs:
#     x_sample = df_debug["aod550"][idx]
#     x_sample = [p for p in np.expand_dims(np.array(x_sample),1)]
#     x_sample = np.array([c.astype("float32") for c in x_sample])[0,0]
#     samples.append(x_sample)
#     plt.imshow(np.array(x_sample))
#     plt.show()


# 2*samples[1]-samples[2]










