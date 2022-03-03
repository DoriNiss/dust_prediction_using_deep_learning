#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/')
from utils.files_loading import *
from utils.data_exploration import *
from utils.meteorology_printing import *

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import matplotlib.pyplot as plt


get_ipython().system(' ls ../../data/dust_61368_108_2_339')


data_dir = "../../data"
dust_dir = f"{data_dir}/dust_61368_108_2_339"
dust_full_tensor_filename = f"{dust_dir}/dust_61368_108_2_339_full_tensor.pkl"
dust_full_timestamps_filename = f"{dust_dir}/dust_61368_108_2_339_full_timestamps.pkl"
metadata_filename = f"{dust_dir}/dust_61368_108_2_339_metadata.pkl"


metadata = torch.load(metadata_filename)
stations_dict = metadata['idxs']['dims'][1]


stations_metadata_xy_dict = {}
for pm in ["PM10","PM25"]:
    for s in metadata[pm]:
        if 'Name' not in s.keys(): continue
        if s['Name'] in stations_metadata_xy_dict.keys(): continue
        s_idx = list(stations_dict.keys())[list(stations_dict.values()).index(s['Name'])]
        x,y = s['X_ITM'],s['Y_ITM']
        try:
            stations_metadata_xy_dict[s['Name']] = [s_idx,int(x),int(y)]
        except:
            stations_metadata_xy_dict[s['Name']] = [s_idx,np.nan,np.nan]
stations_metadata_xy_dict,len(stations_metadata_xy_dict.keys())


coordinates = np.array(list(stations_metadata_xy_dict.values()))
coordinates


dtype = [('station_idx', float), ('x', float), ('y', float)]
values = [(v[0],v[1],v[2]) for v in coordinates]
a = np.array(values, dtype=dtype)  
a


coordinates_north_sorted = np.sort(a,axis=0,order=['y'])
coordinates_north_sorted = np.array([[v[0],v[1],v[2]] for v in coordinates_north_sorted])
coordinates_north_sorted


# Defining dust event: 50% of all stations raise above 2 sigma of summer 





dust = torch.load(dust_full_tensor_filename)
timestamps = torch.load(dust_full_timestamps_filename)
print(dust.shape, len(timestamps))


summer_idxs = (timestamps.month==6)+(timestamps.month)==7+(timestamps.month==8)


timestamps[summer_idxs][0]


timestamps


dust_pm10_0_summer = dust[summer_idxs,:,0,0]
dust_pm10_0_summer.shape


stations_means = [0]*108
stations_stds = [0]*108
stations_th = [0]*108
for s_idx in range(108):
    station_valid_dust_idxs = dust_pm10_0_summer[:,s_idx]>0
    station_mean_from_raw_tensor = dust_pm10_0_summer[station_valid_dust_idxs,s_idx].mean(0)
    station_std_from_raw_tensor = dust_pm10_0_summer[station_valid_dust_idxs,s_idx].std(0)
    if station_mean_from_raw_tensor>0:
        stations_means[s_idx]+=station_mean_from_raw_tensor
        stations_stds[s_idx]+=station_std_from_raw_tensor
        stations_th[s_idx]+=station_mean_from_raw_tensor+2*station_std_from_raw_tensor


for i in range(108):
    print(f"{stations_dict[i]}: mean={stations_means[i]},std={stations_stds[i]},th={stations_th[i]}")


# Thresholds -different than paper's, not looking at 24 hourly averages but on 3 hourly!










