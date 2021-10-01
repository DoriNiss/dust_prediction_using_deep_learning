#!/usr/bin/env python
# coding: utf-8

import json
import glob
from PIL import Image
from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
from tqdm.notebook import tqdm
from skimage.draw import random_shapes
# from netCDF4 import Dataset as netDataset
import csv
import cartopy.crs as ccrs
import glob
import pickle
import random


import sys
sys.path.insert(1, '../Packages/data_dust_9_params')

from Chemfarm_loader import * 


# dates = [[y for y in range(2000,2019)],["*"],["*"],["00","06","12","18"]]
dates = [[2003],[12],[16,18],["12"]]
loader1 = Chemfarm_loader("era5/isn", dates)


loader1.print_info()


loader1.set_param("PV")


times = np.array([0])
levs = np.array([2,3,4,5,6,7,8])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(360,441,1) # 0 to 40
loader1.set_idxs([times,levs,lats,lons])


sample_PV, sample_PV_timestamps = loader1.load_data(average_levs=True)


sample_PV.shape


dates = [[2003],[12],[16,18],["12"]]
loader1 = Chemfarm_loader("era5", dates)


loader1.set_param("SLP")


times = np.array([0])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(360,441,1) # 0 to 40
loader1.set_idxs([times,lats,lons])


sample_SLP, sample_SLP_timestamps = loader1.load_data()
sample_SLP.shape











dates = [[y for y in range(2000,2019)],["*"],["*"],["00","06","12","18"]]
loader_PV = Chemfarm_loader("era5/isn", dates)
loader_PV.set_param("PV")
loader_SLP = Chemfarm_loader("era5", dates)
loader_SLP.set_param("SLP")
times = np.array([0])
levs = np.array([2,3,4,5,6,7,8])
lats = np.arange(220,301,1) # 20 to 60
lons = np.arange(360,441,1) # 0 to 40
loader_PV.set_idxs([times,levs,lats,lons])
loader_SLP.set_idxs([times,lats,lons])


PV, PV_timestamps = loader_PV.load_data(average_levs=True)
SLP, SLP_timestamps = loader_SLP.load_data()


PV.shape, SLP.shape





len(PV_timestamps), PV_timestamps[0].print(), PV_timestamps[-1].print()


len(SLP_timestamps), SLP_timestamps[0].print(), SLP_timestamps[-1].print()


# torch.save(PV, "/work/dorini/dust_project/Data/era5/PV_avg310to340_lat_20_40_lon_0_40_data.pkl")
# torch.save(PV_timestamps, "/work/dorini/dust_project/Data/era5/PV_avg310to340_lat_20_40_lon_0_40_timestamps.pkl")
# torch.save(SLP, "/work/dorini/dust_project/Data/era5/SLP_lat_20_40_lon_0_40_data.pkl")
# torch.save(SLP_timestamps, "/work/dorini/dust_project/Data/era5/SLP_lat_20_40_lon_0_40_timestamps.pkl")


torch.save(PV, "/work/dorini/dust_project/Data/modules/era5/PV_avg310to340_2000to2018_lat_20_40_lon_0_40_data.pkl")
torch.save(PV_timestamps, "/work/dorini/dust_project/Data/modules/era5/PV_avg310to340_2000to2018_lat_20_40_lon_0_40_timestamps.pkl")
torch.save(SLP, "/work/dorini/dust_project/Data/modules/era5/SLP_2000to2018_lat_20_40_lon_0_40_data.pkl")
torch.save(SLP_timestamps, "/work/dorini/dust_project/Data/modules/era5/SLP_2000to2018_lat_20_40_lon_0_40_timestamps.pkl")




























