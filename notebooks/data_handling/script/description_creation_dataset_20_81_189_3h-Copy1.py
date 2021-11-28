#!/usr/bin/env python
# coding: utf-8

import torch


description = {
    "input": {
        0: {"short":"SLP", "long":"Mean Sea-Level Pressure"},
        1: {"short":"Z@850", "long":"Geopotential Height at P=850 mBar [m]"},
        2: {"short":"Z@500", "long":"Geopotential Height at P=500 mBar [m]"},
        3: {"short":"Z@250", "long":"Geopotential Height at P=250 mBar [m]"},
        4: {"short":"U@850", "long":"U Component (Eastward) of Wind at P=850 mBar [m/s]"},
        5: {"short":"U@500", "long":"U Component (Eastward) of Wind at P=500 mBar [m/s]"},
        6: {"short":"U@250", "long":"U Component (Eastward) of Wind at P=250 mBar [m/s]"},
        7: {"short":"V@850", "long":"V Component (Northward) of Wind at P=850 mBar [m/s]"},
        8: {"short":"V@500", "long":"V Component (Northward) of Wind at P=500 mBar [m/s]"},
        9: {"short":"V@250", "long":"V Component (Northward) of Wind at P=250 mBar [m/s]"},
        10: {"short":"PV@325", "long":"Potential Vorticity at T=325K [pvu]"},
        11: {"short":"PV@330", "long":"Potential Vorticity at T=330K [pvu]"},
        12: {"short":"PV@335", "long":"Potential Vorticity at T=335K [pvu]"},
        13: {"short":"PV@340", "long":"Potential Vorticity at T=340K [pvu]"},
        14: {"short":"aod550", "long":"Total Aerosol Optical Depth at 550nm"},
        15: {"short":"duaod550", "long":"Dust Aerosol Optical Depth at 550nm"},
        16: {"short":"aermssdul", "long":"Vertically Integrated Mass of Dust Aerosol (9 - 20 um)"},
        17: {"short":"aermssdum", "long":"Vertically Integrated Mass of Dust Aerosol (0.55 - 9 um)"},
        18: {"short":"u10", "long":"10 Metre U Wind Component [m/s]"},
        19: {"short":"v10", "long":"10 Metre V Wind Component [m/s]"},
        "lons": "-44:50",
        "lats": "20:60",
        "general": "frequency: 3h, all CAMS (cols [14:]) are linearly interpolated between 6h",
    },
    "target": {}
}

description["target"]["general"] = "PM10 Averages, Half-Hourly Measured and 3-Hourly Averaged"

lags_hours = [0, 24, 48, 72, 96, 120, 144, 168]
for i in range(0,8):
    lag = lags_hours[i]
    description["target"][i] = {"df_col":"dust_"+str(lags_hours[i]), "description": "Dust at T+"
                                +str(lags_hours[i])+"h"}
for i in range(8,16):
    lag = lags_hours[i-8]
    lag_m3_str = f"+{lag-3}" if lag>=3 else f"-{3-lag}"
    description["target"][i] = {"df_col": f"delta_{lag}", 
                                "description": f"Delta Dust at T+{lag}h: Dust(T+{lag}h)-Dust(T{lag_m3_str}h)"}


description





description_path = "../../data/meteorology_dataframes_20_81_189_3h/metadata/meteorology_dataframes_20_81_189_3h_description.pkl"
torch.save(description,description_path)


dir_path = "../../data/datasets_20_81_189_3h_7days_future/metadata"
base_filename_dataset = "dataset_20_81_189_3h_7days_future"
for y in range(2003,2019):
    torch.save(description,f"{dir_path}/{base_filename_dataset}_{y}_descriptions.pkl")




