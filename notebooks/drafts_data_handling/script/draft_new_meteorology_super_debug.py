#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from MeteorologyToPandasHandler_Super import *


meteo_folder = "/work/meteogroup"

params = {}

params["SLP"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"P","netCDF_name":"SLP",
                 "size":[81,189],"hourly_res":3,"title":f"Mean Sea-Level Pressure [hPa]"}

plevs = [900,850,700,500,250]
for plev in plevs:
    params[f"Z{plev}"] =  {"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P","netCDF_name":"Z",
                           "size":[81,189],"hourly_res":3,"title":f"Geopotential Height at {plev}hPa [m]"}
    params[f"U{plev}"] =  {"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P","netCDF_name":"U",
                           "size":[81,189],"hourly_res":3,"title":f"Eastward Wind at {plev}hPa [m/s]"}
    params[f"V{plev}"] =  {"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P","netCDF_name":"V",
                           "size":[81,189],"hourly_res":3,"title":f"Northward Wind at {plev}hPa [m/s]"}
    params[f"W{plev}"] =  {"folder_path":f"{meteo_folder}/erainterim/plev","file_prefix":"P","netCDF_name":"V",
                           "size":[41,95],"hourly_res":6,"title":f"Vertical Wind at {plev}hPa [m/s]"}
    params[f"Q{plev}"] =  {"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P","netCDF_name":"Q",
                           "size":[81,189],"hourly_res":3,"title":f"Specific Humidity at {plev}hPa"}
    params[f"T{plev}"] =  {"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P","netCDF_name":"T",
                           "size":[81,189],"hourly_res":3,"title":f"Air Temperature at {plev}hPa"}
    params[f"PV{plev}"] = {"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P","netCDF_name":"PV",
                           "size":[81,189],"hourly_res":3,"title":f"Potential Vorticity at {plev}hPa [pvu]"}
# clouds:
clouds_model_levs = [80,90,100,110] # up to 131
for lev in clouds_model_levs:
    params[f"CLWC{lev}"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"C","netCDF_name":"CLWC",
                            "size":[81,189],"hourly_res":3,
                            "title":f"Specific Cloud Liquid Water Content at Model Level {lev}"}
    params[f"CIWC{lev}"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"C","netCDF_name":"CIWC",
                            "size":[81,189],"hourly_res":3,
                            "title":f"Specific Cloud Ice Water Content at Model Level {lev}"}
    params[f"CRWC{lev}"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"C","netCDF_name":"CRWC",
                            "size":[81,189],"hourly_res":3,
                            "title":f"Specific Rain Water Content at Model Level {lev}"}
params[f"CLWC_avg"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"C","netCDF_name":"CLWC",
                       "size":[81,189],"hourly_res":3,"title":f"Mean Specific Cloud Liquid Water Content"}
params[f"CIWC_avg"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"C","netCDF_name":"CIWC",
                       "size":[81,189],"hourly_res":3,"title":f"Mean Specific Cloud Ice Water Content"}
params[f"CRWC_avg"] = {"folder_path":f"{meteo_folder}/era5","file_prefix":"C","netCDF_name":"CRWC",
                       "size":[81,189],"hourly_res":3,"title":f"Mean Specific Rain Water Content"}

# cams:
cams_model_levs = [20,30,40,50] # up to 60
for lev in cams_model_levs:
    params[f"aermr06_{lev}"] = {"folder_path":f"{meteo_folder}/cams","file_prefix":"A","netCDF_name":"aermr06",
                                "size":[41,95],"hourly_res":6,
                                "title":f"Dust Aerosol (0.9 - 20 um) Mixing Ratio at Model Level {lev}"}
params[f"tcwv"] =     {"folder_path":f"{meteo_folder}/cams","file_prefix":"D","netCDF_name":"tcwv",
                       "size":[41,95],"hourly_res":6,"title":f"Total Column Water Vapour"}
params[f"aod550"] =   {"folder_path":f"{meteo_folder}/cams","file_prefix":"D","netCDF_name":"aod550",
                       "size":[41,95],"hourly_res":6,"title":f"Total Aerosol Optical Depth at 550nm"}
params[f"duaod550"] = {"folder_path":f"{meteo_folder}/cams","file_prefix":"D","netCDF_name":"duaod550",
                       "size":[41,95],"hourly_res":6,"title":f"Dust Aerosol Optical Depth at 550nm"}
params[f"u10"]      = {"folder_path":f"{meteo_folder}/cams","file_prefix":"D","netCDF_name":"u10",
                       "size":[41,95],"hourly_res":6,"title":f"Eastward Wind at 10m [m/s]"}
params[f"v10"]      = {"folder_path":f"{meteo_folder}/cams","file_prefix":"D","netCDF_name":"v10",
                       "size":[41,95],"hourly_res":6,"title":f"Northward Wind at 10m [m/s]"}
# params[f"aermssdul"]= {"folder_path":f"{meteo_folder}/cams","file_prefix":"D","netCDF_name":"aermssdul",
#                        "size":[41,95],"hourly_res":6,
#                        "title":f"Vertically Integrated Mass of Dust Aerosol (9 - 20 um)"} # NOT IN 2019-2020

params[f"pm10"]     = {"folder_path":f"{meteo_folder}/cams","file_prefix":"PM","netCDF_name":"pm10",
                       "size":[41,95],"hourly_res":6,"title":f"Calculated Particulate Matter, d < 10 um"}
params[f"pm2p5"]    = {"folder_path":f"{meteo_folder}/cams","file_prefix":"PM","netCDF_name":"pm2p5",
                       "size":[41,95],"hourly_res":6,"title":f"Calculated Particulate Matter, d < 2.5 um"}

len(list(params.keys())),params





debug_dates_strs = ["2018-12-31 12:00","2018-12-31 15:00","2018-12-31 18:00",
                    "2018-12-31 21:00","2019-01-01 00:00"]


dates = pd.to_datetime(debug_dates_strs,utc=True)
handler = MeteorologyToPandasHandler_Super(
    params, dates=dates, debug=True, keep_na=False, result_size=[81,189],result_hourly_res=3
)








# IDXS


handler.print_param_info("SLP")
handler.print_param_varaiable_info("SLP","time") 
handler.print_param_varaiable_info("SLP","lat") 
handler.print_param_varaiable_info("SLP","lon") 


lats = list(range(200,281)) # 10 to 50
lons = list(range(296,485)) # -32 to 62
params_basenames = ['SLP']
for param_basename in params_basenames:
    param = f"{param_basename}"
    idxs_dict = {'time':[0],'lat':lats,'lon':lons}
    handler.set_param_idxs(param, idxs_dict, avg_over_idxs=None)


handler.print_param_info("Z850")
handler.print_param_varaiable_info("Z850","time") 
handler.print_param_varaiable_info("Z850","lev") 
handler.print_param_varaiable_info("Z850","lat") 
handler.print_param_varaiable_info("Z850","lon") 


plevs


lev_idxs = [4,6,11,15,20]
lats = list(range(200,281)) # 10 to 50
lons = list(range(296,485)) # -32 to 62
params_basenames = ['Z','U','V','Q','T','PV']
for param_basename in params_basenames:
    for i,plev in enumerate(plevs):
        param,plev = f"{param_basename}{plev}",lev_idxs[i]
        idxs_dict = {'time':[0],'lev':[plev],'lat':lats,'lon':lons}
        handler.set_param_idxs(param, idxs_dict, avg_over_idxs=None)


# handler.print_param_info("W850")
handler.print_param_varaiable_info("W850","time") 
handler.print_param_varaiable_info("W850","lev") 
handler.print_param_varaiable_info("W850","lat") 
handler.print_param_varaiable_info("W850","lon") 


plevs


lev_idxs = [4,6,11,15,20]
lats = list(range(100,141)) # 10 to 50
lons = list(range(148,243)) # -32 to 62
params_basenames = ['W']
for param_basename in params_basenames:
    for i,lev in enumerate(plevs):
        param,lev = f"{param_basename}{lev}",lev_idxs[i]
        idxs_dict = {'time':[0],'lev':[lev],'lat':lats,'lon':lons}
        handler.set_param_idxs(param, idxs_dict, avg_over_idxs=None)


handler.print_param_info("CLWC80")
handler.print_param_varaiable_info("CLWC80","time") 
handler.print_param_varaiable_info("CLWC80","lev") 
handler.print_param_varaiable_info("CLWC80","lat") 
handler.print_param_varaiable_info("CLWC80","lon") 


clouds_model_levs


lev_idxs = [73,83,93,103] 
lev_idxs_avgs = lev_idxs#list(range(131)) # FOR DEBUGGING!
lats = list(range(200,281)) # 10 to 50
lons = list(range(296,485)) # -32 to 62
params_basenames = ['CLWC','CIWC','CRWC']
for param_basename in params_basenames:
    for i,lev in enumerate(clouds_model_levs):
        param,lev = f"{param_basename}{lev}",lev_idxs[i]
        idxs_dict = {'time':[0],'lev':[lev],'lat':lats,'lon':lons}
        handler.set_param_idxs(param, idxs_dict, avg_over_idxs=None)
        param = f"{param_basename}_avg"
        idxs_dict = {'time':[0],'lev':lev_idxs_avgs,'lat':lats,'lon':lons}
        handler.set_param_idxs(param, idxs_dict, avg_over_idxs="lev")


handler.print_param_info("aermr06_20")
handler.print_param_varaiable_info("aermr06_20","time") 
handler.print_param_varaiable_info("aermr06_20","level") 
handler.print_param_varaiable_info("aermr06_20","latitude") 
handler.print_param_varaiable_info("aermr06_20","longitude") 


cams_model_levs


lev_idxs = [19,29,39,49]
lats = list(range(80,39,-1)) # 10 to 50
lons = list(range(148,243)) # -32 to 62
params_basenames = ['aermr06']
for param_basename in params_basenames:
    for i,lev in enumerate(cams_model_levs):
        param,lev = f"{param_basename}_{lev}",lev_idxs[i]
        idxs_dict = {'time':[0],'level':[lev],'latitude':lats,'longitude':lons}
        handler.set_param_idxs(param, idxs_dict, avg_over_idxs=None)


# handler.print_param_info("tcwv")
handler.print_param_varaiable_info("aod550","time")
handler.print_param_varaiable_info("aod550","latitude")
handler.print_param_varaiable_info("aod550","longitude")


lats = list(range(80,39,-1)) # 10 to 50
lons = list(range(148,243)) # -32 to 62
params_basenames = ['aod550','duaod550','tcwv','u10','v10']
for param_basename in params_basenames:
    param = f"{param_basename}"
    idxs_dict = {'time':[0],'latitude':lats,'longitude':lons}
    handler.set_param_idxs(param, idxs_dict, avg_over_idxs=None)


# handler.print_param_info("pm10")
handler.print_param_varaiable_info("pm10","time")
handler.print_param_varaiable_info("pm10","latitude")
handler.print_param_varaiable_info("pm10","longitude")


lats = list(range(80,39,-1)) # 10 to 50
lons = list(range(148,243)) # -32 to 62
params_basenames = ['pm10','pm2p5']
for param_basename in params_basenames:
    param = f"{param_basename}"
    idxs_dict = {'time':[0],'latitude':lats,'longitude':lons}
    handler.set_param_idxs(param, idxs_dict, avg_over_idxs=None)











len(handler.params_idxs.keys())


base_filename_debug = "../../data/meteorology_dataframes_62_81_189/debug/df_debug"
handler.create_yearly_dataframes_parallel(base_filename_debug,years=None,add_year_to_name=True,njobs=1)





df_2018 = torch.load(f'{base_filename_debug}_2018.pkl')
df_2019 = torch.load(f'{base_filename_debug}_2019.pkl')


df_2018


df_2018["pm10"][0].shape


df_2018.columns


for lev in clouds_model_levs:
    print(df_2018[f"CIWC{lev}"][2][20,30])
print(df_2018["CIWC_avg"][2][50,30])


df_2018["CRWC_avg"][0].shape


df_2018["W900"][1]


df_2018["W900"][0].shape


df_2019.columns


df_2019




























