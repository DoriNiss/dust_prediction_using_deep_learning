#!/usr/bin/env python
# coding: utf-8

import torch


meteorology_description = { # [:,0=<C<17,:,:]
    "idxs": [i for i in range(17)],
    "description": "Meteorological data from ERA5",
    "titles_long": [
        "Sea Level Pressure (SLP) [hPa]", # 0
        "Geopotential Height (Z) at p=850hPa [m]", # 1
        "Geopotential Height (Z) at p=500hPa [m]", # 2
        "Geopotential Height (Z) at p=250hPa [m]", # 3
        "Eastward Wind (U) at p=850hPa [m/s]", # 4
        "Eastward Wind (U) at p=500hPa [m/s]", # 5
        "Eastward Wind (U) at p=250hPa [m/s]", # 6
        "Northward Wind (V) at p=850hPa [m/s]", # 7
        "Northward Wind (V) at p=500hPa [m/s]", # 8
        "Northward Wind (V) at p=250hPa [m/s]", # 9
        "Potential Vorticity (PV) at T=310K [pvu]", # 10
        "Potential Vorticity (PV) at T=315K [pvu]", # 11
        "Potential Vorticity (PV) at T=320K [pvu]", # 12
        "Potential Vorticity (PV) at T=325K [pvu]", # 13
        "Potential Vorticity (PV) at T=330K [pvu]", # 14
        "Potential Vorticity (PV) at T=335K [pvu]", # 15
        "Potential Vorticity (PV) at T=340K [pvu]", # 16
    ],
    "titles_short": [
        "SLP", # 0
        "Z@850", # 1
        "Z@500", # 2
        "Z@250", # 3
        "U@850", # 4
        "U@500", # 5
        "U@250", # 6
        "V@850", # 7
        "V@500", # 8
        "V@250", # 9
        "PV@310", # 10
        "PV@315", # 11
        "PV@320", # 12
        "PV@325", # 13
        "PV@330", # 14
        "PV@335", # 15
        "PV@340", # 16
    ],
    "lats": [n/2. for n in range(20*2,60*2+1)],
    "lons": [n/2. for n in range(0*2,40*2+1)],
}


# len(meteorology_descriptions["lons"]), meteorology_descriptions["lons"]


dust_description = {
    "idxs": [i for i in range(10)],
    "description": "Dust levels at Beer Sheva. dust_n is the PM10 levels of T + n hours, where T is the time of row (given in times file). delta_n is the difference dust_(n+3)-dust_n",
    "titles": [
        "Dust at T", # 0
        "Delta dust at T (dust @ T+3h)-(dust @ T)", # 1
        "Dust at T-24h", # 2
        "Delta dust at T-24h (dust @ T-24h)-(dust @ T-27h)", # 3
        "Dust at T+24h", # 4
        "Delta dust at T+24h (dust @ T+24h)-(dust @ T+21h)", # 5
        "Dust at T+48h", # 6
        "Delta dust at T+48h (dust @ T+48h)-(dust @ T+45h)", # 7
        "Dust at T+72h", # 8
        "Delta dust at T+72h (dust @ T+72h)-(dust @ T+69)", # 9
    ],
}


dir_path = "../../data/metadata_meteo20000101to20210630_dust_0_m24_24_48_72/"
dust_description_path = dir_path+"dust_description.pkl"
meteorology_description_path = dir_path+"meteorology_description.pkl"
torch.save(meteorology_description,meteorology_description_path)
torch.save(dust_description,dust_description_path)

