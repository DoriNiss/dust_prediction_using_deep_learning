#!/usr/bin/env python
# coding: utf-8

import glob
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch
from scipy import interpolate
from joblib import Parallel, delayed #conda install -c anaconda joblib
import scipy.ndimage
from PIL import Image

import sys
sys.path.insert(0, '../../packages/data_handlers')
# from MeteorologyToPandasHandler import *


""" 
Comments from meeting:
CAMS
A: 60 closest to ground (hybrid modal levels) 3D (daod, aod)
PM: (calculated based on daod, aod)
D: aod, duaod, tcwv


ERA5
model 137 closest to ground to 100 (100 can be 5000m), denser next to ground
T,Q specific humidity - 137 to 100 (ERA5)
PV - take from S (ERA5)

precipitation: (not really good)
era5_1h_2d - R files
L - large scale,

cloud: (not really good) ???? complicated
ERA5: C files - take CLWC, CRWC (maybe ice also to get uncertainty?)
relative humidity - s file, specific is better
"""


"""
Taking:
era5:
    C: clouds (multiple heights, sum)
        CLWC - Specific cloud liquid water content 
        CIWC - Specific cloud ice water content 
        CRWC - Specific rain water content
    S:  PV - Potential vorticity
    P:
        T - Air temperature
        Q - Specific humidity
        SLP - Mean sea-level pressure
    plev P:
        U,V,Z 
cams:
    A:  aermr06 - Dust Aerosol (0.9 - 20 um) Mixing Ratio 
    D: 
        tcwv - Total column water vapour
        aod550 - Total Aerosol Optical Depth at 550nm    
        duaod550 - Dust Aerosol Optical Depth at 550nm        
        u10,v10
    PM:
        pm10 - Particulate matter d < 10 um
        pm2p5 - Particulate matter d < 2.5 um

~45 channels
"""


get_ipython().system('ls /work/meteogroup/era5/2020/01')


sample_path = "/work/meteogroup/era5/2020/01/C20200101_00"
sample_file = Dataset(sample_path)
sample_file.variables


values = sample_file["lev"][:].data
print([f"{value} [{i}]" for i,value in enumerate(values)])


sample_path = "/work/meteogroup/era5/2020/01/S20200101_00"
sample_file = Dataset(sample_path)
sample_file.variables


sample_path = "/work/meteogroup/era5/2020/01/P20200101_00"
sample_file = Dataset(sample_path)
sample_file.variables


sample_path = "/work/meteogroup/era5/2020/01/Z20200101_00"
sample_file = Dataset(sample_path)
sample_file.variables


sample_path = "/work/meteogroup/era5/plev/2020/01/P20200101_00"
sample_file = Dataset(sample_path)
sample_file.variables


get_ipython().system('ls /work/meteogroup/erainterim/plev/2018/01')


sample_path = "/work/meteogroup/erainterim/plev/2018/01/P20180101_00"
sample_file = Dataset(sample_path)
sample_file.variables


get_ipython().system('ls /work/meteogroup/cams/2020/01')


sample_path = "/work/meteogroup/cams/2003/01/A20030101_00"
sample_file = Dataset(sample_path)
sample_file.variables


sample_path = "/work/meteogroup/cams/2020/01/A20200101_00"
sample_file = Dataset(sample_path)
sample_file.variables


sample_path = "/work/meteogroup/cams/2020/01/D20200101_00"
sample_file = Dataset(sample_path)
sample_file.variables


sample_path = "/work/meteogroup/cams/2020/01/PM20200101_00"
sample_file = Dataset(sample_path)
sample_file.variables














start,end = "2020-12-31 00:00","2020-12-31 23:00"
result_hourly_res = 3
freq = f"{result_hourly_res}h"
pd.date_range(start=start, end=end, tz="UTC", freq=freq), freq


debug = False
start,end = ("2002-12-30 00:00","2003-01-02 18:00") if debug else ("2000-01-01 00:00","2020-12-31 23:00") 
start


debug = True
if debug: start,end = ("2002-12-30 00:00","2003-01-02 18:00")
else: start,end = ("2000-01-01 00:00","2020-12-31 23:00") 
start,end


a1 = [1,3,4]
a2 = a1.copy()
a2.remove(1)
a2


def init_params_to_take_from_same_paths(params_dict):
    params_to_take_from_same_paths = []
    params_done,params_left = [],list(params_dict.keys())
    for param in params_dict.keys():
        if param in params_done: continue
        params_from_same_path = [param]
        folder_path,file_prefix = params_dict[param]["folder_path"],params_dict[param]["file_prefix"]
        params_left.remove(param)        
        params_done.append(param)
        for other_param in params_left:
            if folder_path==params_dict[other_param]["folder_path"] and file_prefix==params_dict[other_param]["file_prefix"]:
                params_from_same_path.append(other_param)
                params_left.remove(other_param)        
                params_done.append(other_param)
        params_to_take_from_same_paths.append({
            "folder_path":folder_path,"file_prefix":file_prefix,"params":params_from_same_path})
    return params_to_take_from_same_paths


params = {
    "PV310": {"folder_path":"era5","file_prefix":"S"},
    "PV340": {"folder_path":"era5","file_prefix":"S"},
    "T310": {"folder_path":"era5","file_prefix":"P"},
    "Q310": {"folder_path":"era5","file_prefix":"P"},
    "Z500": {"folder_path":"era5/plev","file_prefix":"P"},
    "Z850": {"folder_path":"era5/plev","file_prefix":"P"}
}

init_params_to_take_from_same_paths(params)

























