#!/usr/bin/env python
# coding: utf-8

import torch


description = {
    "input": {
        0: {"short":"SLP", "long":"Mean sea-level pressure"},
        1: {"short":"Z@850", "long":"Geopotential Height at P=850 mBar [m]"},
        2: {"short":"Z@500", "long":"Geopotential Height at P=500 mBar [m]"},
        3: {"short":"Z@250", "long":"Geopotential Height at P=250 mBar [m]"},
        4: {"short":"U@850", "long":"U component (eastward) of wind at P=850 mBar [m/s]"},
        5: {"short":"U@500", "long":"U component (eastward) of wind at P=500 mBar [m/s]"},
        6: {"short":"U@250", "long":"U component (eastward) of wind at P=250 mBar [m/s]"},
        7: {"short":"V@850", "long":"V component (northward) of wind at P=850 mBar [m/s]"},
        8: {"short":"V@500", "long":"V component (northward) of wind at P=500 mBar [m/s]"},
        9: {"short":"V@250", "long":"V component (northward) of wind at P=250 mBar [m/s]"},
#         10: {"short":"PV@310", "long":"Potential Vorticity at T=310K [pvu]"}, # Problematic - bad corner
#         11: {"short":"PV@315", "long":"Potential Vorticity at T=315K [pvu]"}, # Problematic - bad corner
#         12: {"short":"PV@320", "long":"Potential Vorticity at T=320K [pvu]"}, # Problematic - bad corner
        10: {"short":"PV@325", "long":"Potential Vorticity at T=325K [pvu]"},
        11: {"short":"PV@330", "long":"Potential Vorticity at T=330K [pvu]"},
        12: {"short":"PV@335", "long":"Potential Vorticity at T=335K [pvu]"},
        13: {"short":"PV@340", "long":"Potential Vorticity at T=340K [pvu]"},
        14: {"short":"aod550", "long":"Total Aerosol Optical Depth at 550nm"},
        15: {"short":"duaod550", "long":"Dust Aerosol Optical Depth at 550nm"},
        16: {"short":"aermssdul", "long":"Vertically integrated mass of dust aerosol (9 - 20 um)"},
        17: {"short":"aermssdum", "long":"Vertically integrated mass of dust aerosol (0.55 - 9 um)"},
        18: {"short":"u10", "long":"10 metre U wind component [m/s]"},
        19: {"short":"v10", "long":"10 metre V wind component [m/s]"},
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
    description["target"][i+58] = {"df_col":"delta_"+str(i*6), "description": "delta dust at i+"+str(i*6)+"h"}
for i in range(29,57):
    description["target"][i+58] = {"df_col":"delta_m"+str((57-i)*6), "description": "delta dust at i-"+str((57-i)*6)+"h"}


lags = [i for i in range(0,169,6)] + [-i for i in range(168,0,-6)]
print(lags[29:2*29])


description


description_path = "../../data/meteorology_dataframes_23_81_189/metadata/meteorology_dataframes_20_81_189_description.pkl"
torch.save(description,description_path)







