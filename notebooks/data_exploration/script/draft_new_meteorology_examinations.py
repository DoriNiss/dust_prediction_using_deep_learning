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
from data_handlers.DatasetHandler_DataframeToTensor_Meteorology import *
from utils.meteorology_printing import *


# !ls ../../data/meteorology_tensors_62_81_189/


# meteo_folder = "/work/meteogroup"

# params = {}

# params["SLP"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"P","netCDF_name":"SLP",
#                  "size":[81,189],"hourly_res":3,"title":f"Mean Sea-Level Pressure [hPa]"}

# plevs = [900,850,700,500,250]
# for plev in plevs:
#     params[f"Z{plev}"] =  {"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P","netCDF_name":"Z",
#                            "size":[81,189],"hourly_res":3,"title":f"Geopotential Height at {plev}hPa [m]"}
#     params[f"U{plev}"] =  {"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P","netCDF_name":"U",
#                            "size":[81,189],"hourly_res":3,"title":f"Eastward Wind at {plev}hPa [m/s]"}
#     params[f"V{plev}"] =  {"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P","netCDF_name":"V",
#                            "size":[81,189],"hourly_res":3,"title":f"Northward Wind at {plev}hPa [m/s]"}
#     params[f"W{plev}"] =  {"folder_path":f"{meteo_folder}/erainterim/plev","file_prefix":"P","netCDF_name":"V",
#                            "size":[41,95],"hourly_res":6,"title":f"Vertical Wind at {plev}hPa [m/s]"}
#     params[f"Q{plev}"] =  {"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P","netCDF_name":"Q",
#                            "size":[81,189],"hourly_res":3,"title":f"Specific Humidity at {plev}hPa"}
#     params[f"T{plev}"] =  {"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P","netCDF_name":"T",
#                            "size":[81,189],"hourly_res":3,"title":f"Air Temperature at {plev}hPa"}
#     params[f"PV{plev}"] = {"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P","netCDF_name":"PV",
#                            "size":[81,189],"hourly_res":3,"title":f"Potential Vorticity at {plev}hPa [pvu]"}
# # clouds:
# clouds_model_levs = [80,90,100,110] # up to 131
# for lev in clouds_model_levs:
#     params[f"CLWC{lev}"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"C","netCDF_name":"CLWC",
#                             "size":[81,189],"hourly_res":3,
#                             "title":f"Specific Cloud Liquid Water Content at Model Level {lev}"}
#     params[f"CIWC{lev}"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"C","netCDF_name":"CIWC",
#                             "size":[81,189],"hourly_res":3,
#                             "title":f"Specific Cloud Ice Water Content at Model Level {lev}"}
#     params[f"CRWC{lev}"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"C","netCDF_name":"CRWC",
#                             "size":[81,189],"hourly_res":3,
#                             "title":f"Specific Rain Water Content at Model Level {lev}"}
# params[f"CLWC_avg"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"C","netCDF_name":"CLWC",
#                        "size":[81,189],"hourly_res":3,"title":f"Mean Specific Cloud Liquid Water Content"}
# params[f"CIWC_avg"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"C","netCDF_name":"CIWC",
#                        "size":[81,189],"hourly_res":3,"title":f"Mean Specific Cloud Ice Water Content"}
# params[f"CRWC_avg"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"C","netCDF_name":"CRWC",
#                        "size":[81,189],"hourly_res":3,"title":f"Mean Specific Rain Water Content"}

# # cams:
# cams_model_levs = [20,30,40,50] # up to 60
# for lev in cams_model_levs:
#     params[f"aermr06_{lev}"] = {"folder_path":f"{meteo_folder}/cams","file_prefix":"A","netCDF_name":"aermr06",
#                                 "size":[41,95],"hourly_res":6,
#                                 "title":f"Dust Aerosol (0.9 - 20 um) Mixing Ratio at Model Level {lev}"}
# params[f"tcwv"] =     {"folder_path":f"{meteo_folder}/cams","file_prefix":"D","netCDF_name":"tcwv",
#                        "size":[41,95],"hourly_res":6,"title":f"Total Column Water Vapour"}
# params[f"aod550"] =   {"folder_path":f"{meteo_folder}/cams","file_prefix":"D","netCDF_name":"aod550",
#                        "size":[41,95],"hourly_res":6,"title":f"Total Aerosol Optical Depth at 550nm"}
# params[f"duaod550"] = {"folder_path":f"{meteo_folder}/cams","file_prefix":"D","netCDF_name":"duaod550",
#                        "size":[41,95],"hourly_res":6,"title":f"Dust Aerosol Optical Depth at 550nm"}
# params[f"u10"]      = {"folder_path":f"{meteo_folder}/cams","file_prefix":"D","netCDF_name":"u10",
#                        "size":[41,95],"hourly_res":6,"title":f"Eastward Wind at 10m [m/s]"}
# params[f"v10"]      = {"folder_path":f"{meteo_folder}/cams","file_prefix":"D","netCDF_name":"v10",
#                        "size":[41,95],"hourly_res":6,"title":f"Northward Wind at 10m [m/s]"}

# params[f"pm10"]     = {"folder_path":f"{meteo_folder}/cams","file_prefix":"PM","netCDF_name":"pm10",
#                        "size":[41,95],"hourly_res":6,"title":f"Calculated Particulate Matter, d < 10 um"}
# params[f"pm2p5"]    = {"folder_path":f"{meteo_folder}/cams","file_prefix":"PM","netCDF_name":"pm2p5",
#                        "size":[41,95],"hourly_res":6,"title":f"Calculated Particulate Matter, d < 2.5 um"}
# params


# metadata = torch.load(f"{tensors_dir}/meteorology_tensor_62_81_189_metadata.pkl")
# metadata["params_full"] = params
# torch.save(metadata,f"{tensors_dir}/meteorology_tensor_62_81_189_metadata.pkl")


tensors_dir = "../../data/meteorology_tensors_62_81_189/"
metadata = torch.load(f"{tensors_dir}/meteorology_tensor_62_81_189_metadata.pkl")

fixed_rows_channels = [4,11,18,25,32]
all_zeros_channels = [36,38,41,44]+list(range(51,62))+[37]
bad_channels = list(set(fixed_rows_channels+all_zeros_channels))
lats = list(metadata["dims"]["lats"].values())
lons = list(metadata["dims"]["lons"].values())


[(c,metadata["dims"]["channels"][c]) for c in bad_channels]


sample_year_2003_x = torch.load(f"{tensors_dir}/meteorology_tensor_62_81_189_2003_tensor_full.pkl")
sample_year_2003_timestamps = torch.load(f"{tensors_dir}/meteorology_tensor_62_81_189_2003_timestamps_full.pkl")
sample_year_2019_x = torch.load(f"{tensors_dir}/meteorology_tensor_62_81_189_2019_tensor_full.pkl")
sample_year_2019_timestamps = torch.load(f"{tensors_dir}/meteorology_tensor_62_81_189_2019_timestamps_full.pkl")


sample_year_2003_x.shape,len(sample_year_2003_timestamps),sample_year_2019_x.shape,len(sample_year_2019_timestamps)


sample_tensors,sample_timestamps = sample_year_2003_x,sample_year_2003_timestamps

idxs_start,idxs_end=300,306
sample_times = sample_timestamps[idxs_start:idxs_end]
sample_channels = [0,4,32,37,44,60]
tensors = [sample_tensors[i,c,:,:] for c in sample_channels for i in range(idxs_start,idxs_end)]
cols_titles = sample_times
rows_titles = [f"{(c,metadata['dims']['channels'][c])}" for c in sample_channels]

print_tensors_with_cartopy(tensors, main_title="", titles=None, num_rows=None, num_cols=6,
                           lons=lons, lats=lats, save_as="",lock_bar=False, lock_bar_idxs=None, 
                           num_levels=None, levels_around_zero=False, manual_levels=None,
                           titles_only_on_edges=True, cols_titles=cols_titles, rows_titles=rows_titles,
                           lock_bar_rows_separately=False)


metadata["params_full"]


cols_titles,rows_titles


metadata["params_full"]["W900"],metadata["params_full"]["pm10"]


get_ipython().system('ls /work/meteogroup/erainterim/plev/2003/02')


get_ipython().system('ls /work/meteogroup/cams/2003/02')


from netCDF4 import Dataset

sample_file = Dataset('/work/meteogroup/erainterim/plev/2003/02/P20030207_12')


sample_file.variables





print("#### lev")
print([f"{i}: {v}" for i,v in enumerate(sample_file.variables["lev"][:].data)])
print("#### lat")
print([f"{i}: {v}" for i,v in enumerate(sample_file.variables["lat"][:].data)])
print("#### lon")
print([f"{i}: {v}" for i,v in enumerate(sample_file.variables["lon"][:].data)])


lats_w = list(range(100,141))
lons_w = list(range(148,243))


print(lats,lons)


sample_w = sample_file["W"][0,4,lats_w,lons_w]
sample_w.shape


plt.imshow(sample_w)


plt.imshow(sample_file["W"][0,11,lats_w,lons_w])


sample_file_pm = Dataset('/work/meteogroup/cams/2003/02/PM20030207_12')


sample_file_pm.variables


print("#### latitude")
print([f"{i}: {v}" for i,v in enumerate(sample_file_pm.variables["latitude"][:].data)])
print("#### longitude")
print([f"{i}: {v}" for i,v in enumerate(sample_file_pm.variables["longitude"][:].data)])


lats_pm = list(range(80,39,-1))
lons_pm = list(range(148,243))
print(lats_pm,lons_pm)


sample_pm = sample_file_pm["pm10"][0,lats_pm,lons_pm]
sample_pm.shape


plt.imshow(sample_pm)


sample_file_w_2013 = Dataset('/work/meteogroup/erainterim/plev/2003/01/P20030101_12')
# sample_file_w_2019 = Dataset('/work/meteogroup/erainterim/plev/2019/02/P20190207_12')
sample_file_pm_2013 = Dataset('/work/meteogroup/cams/2003/01/PM20030101_12')
sample_file_pm_2019 = Dataset('/work/meteogroup/cams/2019/01/PM20190101_12')

sample_w_2013 = sample_file_w_2013["W"][0,4,lats_w,lons_w]
# sample_w_2019 = sample_file_w_2019["W"][0,4,lats_w,lons_w]
sample_pm_2013 = sample_file_pm_2013["pm10"][0,lats_pm,lons_pm]
sample_pm_2019 = sample_file_pm_2019["pm10"][0,lats_pm,lons_pm]

print("sample_w_2013")
plt.imshow(sample_w_2013)
plt.show()
# print("sample_w_2019")
# plt.imshow(sample_w_2019)
# plt.show()
print("sample_pm_2013")
plt.imshow(sample_pm_2013)
plt.show()
print("sample_pm_2019")
plt.imshow(sample_pm_2019)
plt.show()


get_ipython().system('ls ../../data/meteorology_dataframes_62_81_189')


df_dir = "../../data/meteorology_dataframes_62_81_189"
sample_df_2013_path = f"{df_dir}/meteorology_df_2003_b0.pkl"
sample_df_2013 = torch.load(sample_df_2013_path)
sample_df_2013


sample_df_2013["W900"][0]


# There is a bug with the small tensor. Will be debugged in another file!

