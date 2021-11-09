#!/usr/bin/env python
# coding: utf-8

import torch


description = {
    0: {"short":"SLP", "long":"Mean sea-level pressure"},
    1: {"short":"Z@850", "long":"Geopotential Height at P=850 mBar [m]"},
    2: {"short":"Z@500", "long":"Geopotential Height at P=500 mBar [m]"},
    3: {"short":"Z@250", "long":"Geopotential Height at P=250 mBar [m]"},
    4: {"short":"U@1000", "long":"U component (eastward) of wind at P=1000 mBar [m/s]"},
    5: {"short":"U@850", "long":"U component (eastward) of wind at P=850 mBar [m/s]"},
    6: {"short":"U@500", "long":"U component (eastward) of wind at P=500 mBar [m/s]"},
    7: {"short":"U@250", "long":"U component (eastward) of wind at P=250 mBar [m/s]"},
    8: {"short":"V@1000", "long":"V component (northward) of wind at P=1000 mBar [m/s]"},
    9: {"short":"V@850", "long":"V component (northward) of wind at P=850 mBar [m/s]"},
    10: {"short":"V@500", "long":"V component (northward) of wind at P=500 mBar [m/s]"},
    11: {"short":"V@250", "long":"V component (northward) of wind at P=250 mBar [m/s]"},
    12: {"short":"PV@310", "long":"Potential Vorticity at T=310K [pvu]"},
    13: {"short":"PV@315", "long":"Potential Vorticity at T=315K [pvu]"},
    14: {"short":"PV@320", "long":"Potential Vorticity at T=320K [pvu]"},
    15: {"short":"PV@325", "long":"Potential Vorticity at T=325K [pvu]"},
    16: {"short":"PV@330", "long":"Potential Vorticity at T=330K [pvu]"},
    17: {"short":"PV@335", "long":"Potential Vorticity at T=335K [pvu]"},
    18: {"short":"PV@340", "long":"Potential Vorticity at T=340K [pvu]"},
    19: {"short":"aod550", "long":"Total Aerosol Optical Depth at 550nm"},
    20: {"short":"duaod550", "long":"Dust Aerosol Optical Depth at 550nm"},
    21: {"short":"aermssdul", "long":"Vertically integrated mass of dust aerosol (9 - 20 um)"},
    22: {"short":"aermssdum", "long":"Vertically integrated mass of dust aerosol (0.55 - 9 um)"},
}


description_path = "../../data/meteorology_dataframes_23_81_169_keep_na/metadata/meteorology_dataframes_23_81_169_keep_na_description.pkl"
torch.save(description,description_path)




