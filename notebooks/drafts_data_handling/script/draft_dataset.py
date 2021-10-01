#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from Dataset_handler import *

import pandas as pd
import numpy as np
import torch


filename_meteorology = '../../data/meteorology_dataframe_debug_20000101to20210630.pkl'
filename_dust = '../../data/dust_lags_20000101to20203006_0_m24_24_48_72.pkl'


A = []
B = []
C = []
for i in range(5):
    A.append(np.array([i,i+1,i+2]))
    B.append(np.array([10*i,10*i+1,10*i+2]))
    C.append(np.array([[100*i,100*i+1,100*i+2],[110*i,110*i+1,110*i+2]]))
print(A[0].shape,B[0].shape,C[0].shape)


df1 = pd.DataFrame(
    {"A":A,
     "B":B,
     "C":C})

df1


df1[0:1]["C"][0].shape


dataset = Dataset_handler(filename_meteorology=filename_meteorology, filename_dust=filename_dust)


dataset.dataframe.columns[0]


dataset.dataframe["SLP"][0].shape


dataset.dataframe["U"][0].shape


slp_test = np.expand_dims(dataset.dataframe["SLP"][0],0)
slp_test.shape


idx1 = dataset.dataframe.index[0]
idx2 = dataset.dataframe.index[1]
dataset.dataframe["SLP"][idx1]


columns_names = [dataset.dataframe.columns[col_name_idx] for col_name_idx in [0,3,4]]
columns_names


print(slp_test.shape,dataset.dataframe["U"][0].shape)
# test = np.stack([slp_test,dataset.dataframe["U"][0]]) # not working - different sizes
splitted = np.split(dataset.dataframe["U"][0],3,axis=1)
print(splitted[0].shape)
list_for_stacking = [slp_test]+splitted
print("")
for a in list_for_stacking:
    print(a.shape)

np.stack(list_for_stacking,axis=2).squeeze(0).shape


dataset.create_meteorology_tensor(dataset.dataframe).shape


years_2000 = dataset.dataframe[dataset.dataframe.index.year==2000]
years_2000.index


months_6 = dataset.dataframe[dataset.dataframe.index.month==6]
months_7 = dataset.dataframe[dataset.dataframe.index.month==7]
months_6.index, months_7.index


both_6_7 = pd.concat([months_6,months_7])
both_6_7.index


dataset.split_train_valid([2000], [2000])


dataset.train_df.count()


dust_columns_idxs=[5,6,7,8,9,10,11,12,13,14]
dust_lags = [dataset.train_df.columns[column_idx] for column_idx in dust_columns_idxs]
dust_lags


dust_0 = dataset.dataframe['dust_0'].values
delta_0 = dataset.dataframe['delta_0'].values
new_array = np.stack([dust_0,delta_0], axis=1)
new_array.shape


dust = dataset.create_dust_tensor(dataset.dataframe)
dust.shape








data_folder = "../../data/tensors_debug_1/"
dummy_handler_path = data_folder+"dummy_dataset_handler.pkl"
dummy_dataset_handler = Dataset_handler(filename_meteorology, filename_dust, filename_combined=dummy_handler_path)


years_train = [2000]
years_valid = [2000]
dummy_dataset_handler.split_train_valid(years_train,years_valid)


tensors_dict = dummy_dataset_handler.create_datasets(folder_path=data_folder)




