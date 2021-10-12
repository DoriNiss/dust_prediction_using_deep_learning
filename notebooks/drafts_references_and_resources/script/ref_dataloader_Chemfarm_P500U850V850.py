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


dates = [[y for y in range(2000,2016)],["*"],["*"],["00","06","12","18"]]
loader1 = Chemfarm_loader("erainterim/plev_P", dates)


loader1.print_info()


loader1.set_param("Z")


times = np.array([0])
levs = np.array([15])
lats = np.arange(90,161,1)
lons = np.arange(150,241,1)
loader1.set_idxs([times,levs,lats,lons])


Z500, Z500_timestamps = loader1.load_data()


loader1.set_param("U")
times = np.array([0])
levs = np.array([6])
lats = np.arange(90,161,1)
lons = np.arange(150,241,1)
loader1.set_idxs([times,levs,lats,lons])
U850, U850_timestamps = loader1.load_data()
loader1.set_param("V")
V850, V850_timestamps = loader1.load_data()


torch.save(Z500, "../Data/modules/erainterim_Z500_lat_0_70_lon_m30_60_data.pkl")
torch.save(Z500_timestamps, "../Data/modules/erainterim_Z500_lat_0_70_lon_m30_60_timestamps.pkl")
torch.save(U850, "../Data/modules/erainterim_U850_lat_0_70_lon_m30_60_data.pkl")
torch.save(U850_timestamps, "../Data/modules/erainterim_U850_lat_0_70_lon_m30_60_timestamps.pkl")
torch.save(V850, "../Data/modules/erainterim_V850_lat_0_70_lon_m30_60_data.pkl")
torch.save(V850_timestamps, "../Data/modules/erainterim_V850_lat_0_70_lon_m30_60_timestamps.pkl")


V850.shape











# updating timestamps due to Timestamp update... should work the same without this line now


timestamps = loader1.timestamps_from_paths(loader1.paths)


len(timestamps), timestamps[0].print(), timestamps[-1].print()


torch.save(timestamps, "../Data/modules/erainterim_Z500_lat_0_70_lon_m30_60_timestamps.pkl")
torch.save(timestamps, "../Data/modules/erainterim_U850_lat_0_70_lon_m30_60_timestamps.pkl")
torch.save(timestamps, "../Data/modules/erainterim_V850_lat_0_70_lon_m30_60_timestamps.pkl")




