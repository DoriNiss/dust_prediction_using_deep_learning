#!/usr/bin/env python
# coding: utf-8

import torch
import sys
sys.path.insert(0, '../../packages/data_handlers')
from DatasetHandler import *
import numpy as np


years = [y for y in range(2003, 2019)]
meteorology_dataframes_dir = "../../data/meteorology_dataframes_23_81_189/"
dust_dataframes_dir = "../../data/dust_20000101to20213006_6h_7days_before_and_after/"


dataframes_descriptions_paths = [
    "../../data/meteorology_dataframes_23_81_189/metadata/meteorology_dataframes_20_81_189_description.pkl"
]
dataframes_descriptions = [torch.load(path) for path in dataframes_descriptions_paths]


dataframes_paths = []
for y in years:
    dataframes_paths.append([
        meteorology_dataframes_dir+"meteorology_dataframe_23_81_189_"+str(y)+".pkl",
        dust_dataframes_dir+"dust_dataframe_"+str(y)+".pkl",
    ])

        
datasets_arguments = []
lags = [i for i in range(0,169,6)] + [-i for i in range(168,0,-6)]
lags_names = [str(lag) if lag>=0 else "m"+str(-lag) for lag in lags]
for y in years:        
    datasets_arguments.append(
        {
            "cols_input": ["SLP","Z","U","V","PV","aod550","duaod550","aermssdul","aermssdum","u10","v10"],
            "cols_target": ["dust_"+str(lag) for lag in lags_names] + ["delta_"+str(lag) for lag in lags_names],
            "dataframes_descriptions": dataframes_descriptions,
            "keep_na": True,
            "include_all_timestamps_between": True,
            "all_timestamps_intervals": "6h",
            "cols_channels_input": {"PV":[3,4,5,6]},
            "cols_channels_target": None,
            "as_float32": True,
            "wanted_year": y,
            "replace_na_in_target": 0,
        })

        
datasets_dir = "../../data/datasets_20_81_189"
save_as_list = []
for i in range(len(dataframes_paths)):
    save_as_list.append({
        "dir_path": datasets_dir,
        "base_filename": "dataset_20_81_189_"+str(years[i])
    })


# sample_idx = 1
# print("loading",dataframes_paths[sample_idx])
# sample_dataframes = [torch.load(dataframes_paths[sample_idx][i]) for i in range(2)]


# debug_dataframes_path = "../../data/meteorology_dataframes_23_81_189/debug_dataframes.pkl"
# # torch.save([sample_dataframes[0][100:120],sample_dataframes[1][100:120]],debug_dataframes_path)
# debug_dataframes = torch.load(debug_dataframes_path)


# print(debug_dataframes)


# handler = DatasetHandler(debug_dataframes, datasets_arguments[1])


# handler.col_channels_idxs["input"]


# handler.combined_dataframe["PV"][0].shape


# handler.combined_dataframe["PV"][0:1]


# handler.create_tensor_from_dataset_type("input")





DatasetHandler.create_and_save_datasets_from_paths(dataframes_paths, datasets_arguments, save_as_list, njobs=3)








sample_year = 2004
sample_input = torch.load(datasets_dir+"/dataset_20_81_189_"+str(sample_year)+"_input.pkl")
sample_target = torch.load(datasets_dir+"/dataset_20_81_189_"+str(sample_year)+"_target.pkl")
sample_timestamps = torch.load(datasets_dir+"/dataset_20_81_189_"+str(sample_year)+"_timestamps.pkl")


sample_input.shape, sample_target.shape, len(sample_timestamps)


import matplotlib.pyplot as plt
from tqdm import tqdm


idx = 100
for c in range(20):
    print(c)
    plt.imshow(sample_input[idx,c])
    plt.show()


print("Num of NaN's in meteorology data:")
counter = {10:0, 11:0, 12:0}
maxes = {10:{"max":0}, 11:{"max":0}, 12:{"max":0}} 
for y in tqdm(range(2003,2019)):
    x = torch.load(datasets_dir+"/dataset_20_81_189_"+str(y)+"_input.pkl")
    for i in range(x.shape[0]):
        for c in range(20):
            num_nan = torch.sum(torch.isnan(x[i,c])==True)
            if num_nan>0:
                if maxes[c]["max"]<num_nan:
                    maxes[c]={"max":num_nan, "year":y, "idx":i}
                print(y, i, c, num_nan)
                counter[c]+=1
print(counter)
print(maxes)














# Before cutting PV at 310, 315, 320:





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







