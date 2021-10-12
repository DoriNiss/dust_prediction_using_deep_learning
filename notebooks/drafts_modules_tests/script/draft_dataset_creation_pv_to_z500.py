#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from DatasetHandler_TensorToTensor import *
import numpy as np
import pandas as np
import torch


# %pip install --upgrade pandas


data_dir_path = "../../data/"
years_train_split1 = torch.load(data_dir_path+"yearly_splits/split1_train_ordered.pkl")
years_valid_split1 = torch.load(data_dir_path+"yearly_splits/split1_valid_ordered.pkl")
years_train_split2 = torch.load(data_dir_path+"yearly_splits/split2_train_extreme_ratios_in_train_avg_in_valid.pkl")
years_valid_split2 = torch.load(data_dir_path+"yearly_splits/split2_valid_extreme_ratios_in_train_avg_in_valid.pkl")
years_train_split3 = torch.load(data_dir_path+"yearly_splits/split3_train_extreme_ratios_in_valid_avg_in_train.pkl")
years_valid_split3 = torch.load(data_dir_path+"yearly_splits/split3_valid_extreme_ratios_in_valid_avg_in_train.pkl")
years_train_split4 = torch.load(data_dir_path+"yearly_splits/split4_train_max_num_events_in_train_avg_valid.pkl")
years_valid_split4 = torch.load(data_dir_path+"yearly_splits/split4_valid_max_num_events_in_train_avg_valid.pkl")
years_train_split5 = torch.load(data_dir_path+"yearly_splits/split5_train_train_distant_years_valid_between.pkl")
years_valid_split5 = torch.load(data_dir_path+"yearly_splits/split5_valid_train_distant_years_valid_between.pkl")


dataframe_2000 = torch.load(data_dir_path+"pv_to_z500_wide/meteorology_pv_z500_dataframe_2000.pkl")
dataframe_2001 = torch.load(data_dir_path+"pv_to_z500_wide/meteorology_pv_z500_dataframe_2001.pkl")


dataframe_2000["PV"][0].shape


# years_train_split1


dataset_handler = DatasetHandler_TensorToTensor([dataframe_2000, dataframe_2001], ["PV"], ["Z"], debug=True)


dataset_handler = DatasetHandler_TensorToTensor([dataframe_2000, dataframe_2001], ["PV","Z"],["PV"], debug=True)


# dataset_handler = DatasetHandler_TensorToTensor([dataframe_2000, dataframe_2001], ["PV"],["PV","Z"], debug=True)


# dataset_handler = DatasetHandler_TensorToTensor([dataframe_2000, dataframe_2001], ["PV","Z"],["PV","Z"], debug=True)


dataframe_2000["PV"][0].shape


x,y,times = dataset_handler.build_dataset()
x.shape, y.shape, len(times)




