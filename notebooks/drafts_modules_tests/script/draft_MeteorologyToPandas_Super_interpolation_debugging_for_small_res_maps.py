#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import csv
from tqdm import tqdm
from joblib import Parallel, delayed #conda install -c anaconda joblib

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.MeteorologyToPandasHandler_Super import *
from utils.meteorology_printing import *
from data_handlers.DatasetHandler_DataframeToTensor_Meteorology import *


lats_w = list(range(100,141))
lons_w = list(range(148,243))
lats_pm = list(range(80,39,-1))
lons_pm = list(range(148,243))


sample_file_w = Dataset('/work/meteogroup/erainterim/plev/2003/01/P20030101_12')["W"][0,4,lats_w,lons_w]
sample_file_pm = Dataset('/work/meteogroup/cams/2003/01/PM20030101_12')["pm10"][0,lats_pm,lons_pm]


print("Sample W")
plt.imshow(sample_file_w)
plt.show()
print("Sample pm10")
plt.imshow(sample_file_pm)
plt.show()











meteo_folder = "/work/meteogroup"
params = {}
params[f"W900"] =  {"folder_path":f"{meteo_folder}/erainterim/plev","file_prefix":"P","netCDF_name":"W",
                       "size":[41,95],"hourly_res":6,"title":f"Vertical Wind at 900hPa [m/s]"}
params[f"pm10"]     = {"folder_path":f"{meteo_folder}/cams","file_prefix":"PM","netCDF_name":"pm10",
                       "size":[41,95],"hourly_res":6,"title":f"Calculated Particulate Matter, d < 10 um"}
params


dates = pd.to_datetime(["2003-01-01 12:00","2003-01-01 15:00","2003-01-01 18:00"],utc=True)
dates


handler = MeteorologyToPandasHandler_Super(
    params, dates=dates, debug=False, keep_na=False, result_size=[81,189],result_hourly_res=3
)


handler.paths_and_params


handler.print_param_info("W900")
handler.print_param_varaiable_info("W900","time") 
handler.print_param_varaiable_info("W900","lat") 
handler.print_param_varaiable_info("W900","lon") 
handler.print_param_varaiable_info("W900","lev") 


idxs_dict = {'time':[0],'lev':[4],'lat':lats_w,'lon':lons_w}
handler.set_param_idxs('W900',idxs_dict, avg_over_idxs=None)


handler.print_param_info("pm10")
handler.print_param_varaiable_info("pm10","time") 
handler.print_param_varaiable_info("pm10","latitude") 
handler.print_param_varaiable_info("pm10","longitude") 


idxs_dict = {'time':[0],'latitude':lats_pm,'longitude':lons_pm}
handler.set_param_idxs('pm10',idxs_dict, avg_over_idxs=None)





df = handler.create_dataframe(handler.dates[0],handler.dates[-1])
df


(0.06878834+0.015538141)/2


print("Sample W")
plt.imshow(sample_file_w)
plt.show()
print("W from DF")
plt.imshow(df["W900"][0])
plt.show()

print("Sample pm10")
plt.imshow(sample_file_pm)
plt.show()
print("PM10 from DF")
plt.imshow(df["pm10"][0])
plt.show()


# BUG FIXED IN DATAFRAME CREATION!








# Testing dataset creation:


meteo_folder = "/work/meteogroup"
params = {}
params[f"W900"] =  {"folder_path":f"{meteo_folder}/erainterim/plev","file_prefix":"P","netCDF_name":"W",
                       "size":[41,95],"hourly_res":6,"title":f"Vertical Wind at 900hPa [m/s]"}
params[f"aermr06_20"] = {"folder_path":f"{meteo_folder}/cams","file_prefix":"A","netCDF_name":"aermr06",
                         "size":[41,95],"hourly_res":6}
params["SLP"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"P","netCDF_name":"SLP",
                 "size":[81,189],"hourly_res":3,"title":f"Mean Sea-Level Pressure [hPa]"}
params


dates = pd.to_datetime(["2003-01-01 12:00","2003-01-01 15:00","2003-01-01 18:00","2003-01-01 21:00",
                        "2003-07-30 12:00","2019-11-20 09:00","2019-11-20 12:00"],utc=True)
dates


handler = MeteorologyToPandasHandler_Super(
    params, dates=dates, debug=False, keep_na=False, result_size=[81,189],result_hourly_res=3
)


idxs_dict = {'time':[0],'lev':[4],'lat':lats_w,'lon':lons_w}
handler.set_param_idxs('W900',idxs_dict, avg_over_idxs=None)
# handler.print_param_info("aermr06_20")
idxs_dict = {'time':[0],'level':[19],'latitude':lats_pm,'longitude':lons_pm}
handler.set_param_idxs('aermr06_20',idxs_dict, avg_over_idxs=None)


handler.print_param_info("SLP")
handler.print_param_varaiable_info("SLP","time") 
handler.print_param_varaiable_info("SLP","lat") 
handler.print_param_varaiable_info("SLP","lon") 


lats_slp = list(range(200,281)) # 10 to 50
lons_slp = list(range(296,485)) # -32 to 62
idxs_dict = {'time':[0],'lat':lats_slp,'lon':lons_slp}
handler.set_param_idxs('SLP',idxs_dict, avg_over_idxs=None)


df = handler.create_dataframe(handler.dates[0],handler.dates[-1])
df


dims_cols_strings = {1:["W900","aermr06_20","SLP"]}
lats = np.arange(10,50.5,0.5)
lons = np.arange(-32,62.5,0.5)


handler_t = DatasetHandler_DataframeToTensor_Meteorology(
    df, dims_cols_strings=dims_cols_strings, metadata={}, timestamps=None, 
    save_base_filename=None,
    invalid_col_fill=-777, param_shape=[81,189], lons=lons, lats=lats, save_timestamps=False,
    old_format_channels_dict=None, verbose=1
)
t,timestamps,name_t,name_timestampe = handler_t.get_dataset_from_timestamps(df.index, suffix=None)


plt.imshow(t[-1,2].numpy())


handler.print_param_info("aermr06_20")


aer_file = Dataset("/work/meteogroup/cams/2003/01/A20030101_12")["aermr06"][0,19,lats_pm,lons_pm]
aer_file.shape


plt.imshow(aer_file)


plt.imshow(t[0,1].numpy())


aer_file = Dataset("/work/meteogroup/cams/2003/01/A20030101_18")["aermr06"][0,19,lats_pm,lons_pm]
aer_file.shape


plt.imshow(aer_file)


plt.imshow(t[2,1].numpy())


























a = np.array([[-1,-1],[-2,-2]])
a


if a<0:
    print("yes")


a_1 = a.astype("float32")
if a_1<0:
    print("yes")














# Weird file format bug...
# Dataset("/work/meteogroup/cams/2003/07/A20030730_12")





ls /work/meteogroup/cams




