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
    },
    "target": {}
}

for i in range(0,29):
    description["target"][i] = {"df_col":"dust_"+str(i*6), "description": "dust at i+"+str(i*6)+"h"}
for i in range(29,57):
    description["target"][i] = {"df_col":"dust_m"+str((57-i)*6), "description": "dust at i-"+str((57-i)*6)+"h"}
for i in range(0,29):
    description["target"][i+57] = {"df_col":"delta_"+str(i*6), "description": "delta dust at i+"+str(i*6)+"h"}
for i in range(29,57):
    description["target"][i+57] = {"df_col":"delta_m"+str((57-i)*6), "description": "delta dust at i-"+str((57-i)*6)+"h"}


lags = [i for i in range(0,169,6)] + [-i for i in range(168,0,-6)]
print(lags[29:2*29])


description


description_path = "../../data/meteorology_dataframes_23_81_189/metadata/meteorology_dataframes_20_81_189_description.pkl"
torch.save(description,description_path)


import torch
data_dir_original = "../../data/datasets_20_81_189/metadata"
base_filename_original = "dataset_20_81_189"
for y in range(2003,2019):
    torch.save(description,f"{data_dir_original}/{base_filename_original}_{y}_descriptions.pkl")




