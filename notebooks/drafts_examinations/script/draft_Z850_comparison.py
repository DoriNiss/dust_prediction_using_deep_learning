#!/usr/bin/env python
# coding: utf-8

import glob
from netCDF4 import Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np


# from dataset

location = "chemfarm"

if location == "chemfarm":
    dataset_file_path = "../../data/meteorology_dataframes_17_81_81_keep_na/meteorology_df_17_81_81_keep_na_2000.pkl"


sample_df = torch.load(dataset_file_path)
sample_df["Z"][0].shape


sample_idxs = [0,3,6,8,9]
sample_z850 = [sample_df["Z"][i][0,:,:] for i in sample_idxs]
sample_z500 = [sample_df["Z"][i][1,:,:] for i in sample_idxs]
sample_z250 = [sample_df["Z"][i][2,:,:] for i in sample_idxs]
sample_dates = [sample_df.index[i] for i in sample_idxs]





for i in range(len(sample_z850)):
    print(sample_dates[i])
    plt.imshow(sample_z850[i]);
    plt.show();


# sample paths:  glob.glob('/work/meteogroup/era5/plev/2013/12/P2013122[0-9]*')

chemfarm_data_dir = "/work/meteogroup/era5/plev/"
chemfarm_sample_paths = []

def to_chemfarm_str(n):
    return str(n) if n>=10 else "0"+str(n)

for d in sample_dates:
    chemfarm_sample_paths += [chemfarm_data_dir+to_chemfarm_str(d.year)+"/"+to_chemfarm_str(d.month)+"/"+
                              "P"+to_chemfarm_str(d.year)+to_chemfarm_str(d.month)+to_chemfarm_str(d.day)+"_"+
                              to_chemfarm_str(d.hour)]
chemfarm_sample_paths


chemfarm_sample_files = [] 
for p in chemfarm_sample_paths:
    chemfarm_sample_files+=glob.glob(p)
chemfarm_sample_files


sample_ncfile = Dataset(chemfarm_sample_files[0])
print("*** levs:")
for i in range(len(sample_ncfile["lev"])):
    print(i,sample_ncfile["lev"][i].data)
print("\n*** lons:")
for i in range(len(sample_ncfile["lon"])):
    print(i,sample_ncfile["lon"][i])
print("\n*** lats:")
for i in range(len(sample_ncfile["lat"])):
    print(i,sample_ncfile["lat"][i])
    
print(sample_ncfile["Z"].shape)


z850_idx = 6
lon_idxs = np.arange(360,441,1) # 0 to 40
lat_idxs = np.arange(220,301,1) # 20 to 60

print("lev:",sample_ncfile["lev"][z850_idx])
print("lon:",sample_ncfile["lon"][lon_idxs])
print("lat:",sample_ncfile["lat"][lat_idxs])


print("Z850:")
for i,file_path in enumerate(chemfarm_sample_files):
    ncfile = Dataset(file_path) 
    print("chemfarm data:",file_path)
    chemfarm_file = ncfile["Z"][0,z850_idx,lat_idxs,lon_idxs]
    plt.imshow(chemfarm_file);
    plt.show();
    print("dataset:",sample_dates[i])
    plt.imshow(sample_z850[i]);
    plt.show();
    print(chemfarm_file-sample_z850[i])
    


print("Z500:")
for i,file_path in enumerate(chemfarm_sample_files):
    ncfile = Dataset(file_path) 
    print("chemfarm data:",file_path)
    chemfarm_file = ncfile["Z"][0,15,lat_idxs,lon_idxs]
    plt.imshow(chemfarm_file);
    plt.show();
    print("dataset:",sample_dates[i])
    plt.imshow(sample_z500[i]);
    plt.show();
    print(chemfarm_file-sample_z500[i])
    


print("Z250:")
for i,file_path in enumerate(chemfarm_sample_files):
    ncfile = Dataset(file_path) 
    print("chemfarm data:",file_path)
    chemfarm_file = ncfile["Z"][0,20,lat_idxs,lon_idxs]
    plt.imshow(chemfarm_file);
    plt.show();
    print("dataset:",sample_dates[i])
    plt.imshow(sample_z250[i]);
    plt.show();
    print(chemfarm_file-sample_z250[i])








# Correcting with interpolation: (looking at Z850, will use that in the tensor creation step for all params)


bad_z850 = Dataset(chemfarm_sample_files[0])["Z"][0,z850_idx,lat_idxs,lon_idxs]
bad_z850.shape


np.argwhere(np.isnan(bad_z850))


bad_z850[1,35], bad_z850[34,68]





from scipy import interpolate

x = np.arange(0, bad_z850.shape[1])
y = np.arange(0, bad_z850.shape[0])
#mask invalid values
interpolated = np.ma.masked_invalid(bad_z850)
xx, yy = np.meshgrid(x, y)
#get only the valid values
x1 = xx[~interpolated.mask]
y1 = yy[~interpolated.mask]
interpolated = interpolated[~interpolated.mask]

interpolated = interpolate.griddata((x1, y1), interpolated.ravel(),(xx, yy),method='cubic')


plt.imshow(bad_z850)
plt.show();
plt.imshow(interpolated)
plt.show();


np.ma.masked_invalid(bad_z850)[np.ma.masked_invalid(bad_z850).mask].size


np.ma.masked_invalid(interpolated)[np.ma.masked_invalid(interpolated).mask].size




