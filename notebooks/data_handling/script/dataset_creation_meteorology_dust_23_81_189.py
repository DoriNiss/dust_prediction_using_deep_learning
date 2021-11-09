#!/usr/bin/env python
# coding: utf-8

import torch
import sys
sys.path.insert(0, '../../packages/data_handlers')
from DatasetHandler import *
import numpy as np


years = [y for y in range(2000, 2022)]
dataframes_dir = "../../data/meteorology_dataframes_23_81_189/"


dataframes_descriptions_paths = [
    dataframes_dir+"metadata/meteorology_dataframes_23_81_189_description.pkl",
    "../../data/dust_description_pm10_BeerSheva_20000101_20210630_6h.pkl"
]
dataframes_descriptions = [torch.load(path) for path in dataframes_descriptions_paths]


dust_data_path = "../../data/dust_20000101to20213006_6h_7days.pkl"


dataframes_paths = []
for y in years:
    dataframes_paths.append([
        dataframes_dir+"meteorology_dataframe_23_81_189_"+str(y)+".pkl",
        dust_data_path
    ])

        
datasets_arguments = []
for y in years:        
    datasets_arguments.append(
        {
            "cols_input": ["SLP","Z","U","V","PV","aod550","duaod550","aermssdul","aermssdum","u10","v10"],
            "cols_target": ["dust_0","delta_0","dust_m24","delta_m24","dust_24","delta_24",
                            "dust_48","delta_48","dust_72","delta_72","dust_96","delta_96",
                            "dust_120","delta_120","dust_144","delta_144","dust_168","delta_168"],
            "dataframes_descriptions": dataframes_descriptions,
            "keep_na": True,
            "include_all_timestamps_between": True,
            "all_timestamps_intervals": "6h",
            "cols_channels_input": {"SLP": None,"Z": None,"U": None,"V": None,"PV":[3,4,5,6],
                                    "aod550": None,"duaod550": None,"aermssdul": None,"aermssdum": None,
                                    "u10": None,"v10": None}
            "cols_channels_target": None,
            "as_float32": True,
            "wanted_year": y
        })

        
datasets_dir = "../../data/datasets_23_81_189"
save_as_list = []
for i in range(len(dataframes_paths)):
    save_as_list.append({
        "dir_path": datasets_dir,
        "base_filename": "dataset_23_81_189_"+str(years[i])
    })


DatasetHandler.create_and_save_datasets_from_paths(dataframes_paths, datasets_arguments, save_as_list, njobs=3)














sample_year = 2015
sample_input = torch.load(datasets_dir+"/dataset_23_81_189_"+str(sample_year)+"_input.pkl")
sample_target = torch.load(datasets_dir+"/dataset_23_81_189_"+str(sample_year)+"_target.pkl")
sample_timestamps = torch.load(datasets_dir+"/dataset_23_81_189_"+str(sample_year)+"_timestamps.pkl")


sample_input.shape, sample_target.shape, len(sample_timestamps)


import matplotlib.pyplot as plt
from tqdm import tqdm


# idx = 100
# for c in range(23):
#     print(c)
#     plt.imshow(sample_input[idx,c])
#     plt.show()


counter = {10:0, 11:0, 12:0}
maxes = {10:{"max":0}, 11:{"max":0}, 12:{"max":0}} 
for y in tqdm(range(2003,2018)):
    x = torch.load(datasets_dir+"/dataset_23_81_189_"+str(y)+"_input.pkl")
    for i in range(x.shape[0]):
        for c in range(23):
            num_nan = torch.sum(torch.isnan(x[i,c])==True)
            if num_nan>0:
                if maxes[c]["max"]<num_nan:
                    maxes[c]={"max":num_nan, "year":y, "idx":i}
                print(y, i, c, num_nan)
                counter[c]+=1
print(counter)
print(maxes)








print(10)
x = torch.load(datasets_dir+"/dataset_23_81_189_2012_input.pkl")
idx = 743
plt.imshow(x[idx,10])
plt.show()

print(11)
x = torch.load(datasets_dir+"/dataset_23_81_189_2009_input.pkl")
idx = 882
plt.imshow(x[idx,11])
plt.show()

print(12)
# x = torch.load(datasets_dir+"/dataset_23_81_189_2009_input.pkl")
idx = 886
plt.imshow(x[idx,12])
plt.show()


# x = torch.load(datasets_dir+"/dataset_23_81_189_2012_input.pkl")
idx = 743
plt.imshow(x[idx,10,:35,168:])
plt.show()


mask = torch.ones_like(x[0])
mask[10,:35,168:] = 0
plt.imshow(mask[10])


# x = torch.load(datasets_dir+"/dataset_23_81_189_2009_input.pkl")
idx = 882
plt.imshow(x[idx,11,:15,169:])
plt.show()


mask[11,:15,169:] = 0
plt.imshow(mask[11])


# x = torch.load(datasets_dir+"/dataset_23_81_189_2009_input.pkl")
idx = 886
plt.imshow(x[idx,12,:13,170:])
plt.show()


mask[12,:13,170:] = 0
plt.imshow(mask[11])


masked_x = x.clone()
masked_x[torch.isnan(masked_x)]=0
masked_x = masked_x[:]*mask


print(10)
idx = 743
plt.imshow(masked_x[idx,10])
plt.show()

print(11)
idx = 882
plt.imshow(masked_x[idx,11])
plt.show()

print(12)
idx = 886
plt.imshow(masked_x[idx,12])
plt.show()







