#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from Dataset_handler import *
import numpy as np
import pandas as np
import torch


filename_meteorology = '../../data/meteorology_dataframe_20000101to20210630.pkl'
# filename_meteorology = '../../data/meteorology_dataframe_debug_20000101to20210630.pkl'
filename_dust = '../../data/dust_lags_20000101to20213006_0_m24_24_48_72.pkl'
data_folder = "../../data/tensors_meteo20000101to20210630_dust_0_m24_24_48_72/"
dataset_handler_filename = data_folder+"dataset_handler_nometadata.pkl"
# Note: resulting data is in the range 16/06/2000 to 31/12/2019


dataset_handler = Dataset_handler(filename_meteorology, filename_dust, filename_combined=dataset_handler_filename)


# Printing number of events per year to choose 5 splittings of valid/train: (12 years train, 6 years valid)
for i in range(2000,2020):
    years_t,years_v = [i],[0]
    dataset_handler.split_train_valid(years_t,years_v)    


resulting_statistics = {
    2000: {"events": 162 , "clear": 1072 , "total": 1234 , "ratio": 15.11 },
    2001: {"events": 160 , "clear": 2090 , "total": 2250 , "ratio": 7.66  },
    2002: {"events": 228 , "clear": 1636 , "total": 1864 , "ratio": 13.94 },
    2003: {"events": 411 , "clear": 1634 , "total": 2045 , "ratio": 25.15 },
    2004: {"events": 306 , "clear": 1983 , "total": 2289 , "ratio": 15.43 },
    2005: {"events": 217 , "clear": 1838 , "total": 2055 , "ratio": 11.81 },
    2006: {"events": 309 , "clear": 2099 , "total": 2408 , "ratio": 14.72 },
    2007: {"events": 274 , "clear": 2367 , "total": 2641 , "ratio": 11.58 },
    2008: {"events": 278 , "clear": 2207 , "total": 2485 , "ratio": 12.6  },
    2009: {"events": 251 , "clear": 2044 , "total": 2295 , "ratio": 12.28 },
    2010: {"events": 423 , "clear": 2129 , "total": 2552 , "ratio": 19.87 },
    2011: {"events": 282 , "clear": 1349 , "total": 1631 , "ratio": 20.9  },
    2012: {"events": 151 , "clear": 619  , "total": 770  , "ratio": 24.39 },
    2013: {"events": 292 , "clear": 1881 , "total": 2173 , "ratio": 15.52 },
    2014: {"events": 176 , "clear": 1722 , "total": 1898 , "ratio": 10.22 },
    2015: {"events": 198 , "clear": 1707 , "total": 1905 , "ratio": 11.6  },
    2016: {"events": 246 , "clear": 2055 , "total": 2301 , "ratio": 11.97 },
    2017: {"events": 273 , "clear": 1910 , "total": 2183 , "ratio": 14.29 },
    2018: {"events": 278 , "clear": 2074 , "total": 2352 , "ratio": 13.4  },
    2019: {"events": 334 , "clear": 1618 , "total": 1952 , "ratio": 20.64 },
    "total":{"events": 5249 ,"clear": 36034, "total": 41283 ,"ratio": 14.57 }
}


def statistics_of_split(years_list):
    events,clear,total,ratio_avg = 0,0,0,0
    for y in years_list:
        events+=resulting_statistics[y]["events"]
        clear+=resulting_statistics[y]["clear"]
        total+=resulting_statistics[y]["total"]
        ratio_avg+=resulting_statistics[y]["ratio"]
    ratio_avg/=len(years_list)
    print(f"events: {events}, clear: {clear}, total: {total}, ratio_avg: {ratio_avg}, num years: {len(years_list)}")


# split1 : ordered
years_train_split1_ordered = [y for y in range(2000,2012)]
years_valid_split1_ordered = [y for y in range(2013,2019)]
print("Split 1: Ordered")
print("Training:")
statistics_of_split(years_train_split1_ordered)
print("Validation:")
statistics_of_split(years_valid_split1_ordered)


# split2 : extreme_ratios_in_train_avg_in_valid
years_train_split2_extreme_ratios_in_train_avg_in_valid = [
    2001,
    2003,
    2005,
    2007,
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2019,
]
years_valid_split2_extreme_ratios_in_train_avg_in_valid = [
    2000,
    2002,   
    2004,
    2006,
    2017,
    2018,
]

print("Split 2: extreme_ratios_in_train_avg_in_valid")
print("Training:")
statistics_of_split(years_train_split2_extreme_ratios_in_train_avg_in_valid)
print("Validation:")
statistics_of_split(years_valid_split2_extreme_ratios_in_train_avg_in_valid)


# split3 : extreme_ratios_in_valid_avg_in_train
years_train_split3_extreme_ratios_in_valid_avg_in_train = [
    2000,
    2002,
    2004,
    2005,
    2006,
    2008,
    2009,
    2013,
    2015,
    2016,
    2017,
    2018,
]
years_valid_split3_extreme_ratios_in_valid_avg_in_train = [
    2001,
    2003,
    2010,
    2011,
    2012,
    2019
]

print("Split 3: extreme_ratios_in_valid_avg_in_train")
print("Training:")
statistics_of_split(years_train_split3_extreme_ratios_in_valid_avg_in_train)
print("Validation:")
statistics_of_split(years_valid_split3_extreme_ratios_in_valid_avg_in_train)


# split4 : max_num_events_in_train_avg_valid
years_train_split4_max_num_events_in_train_avg_valid = [
    2003,
    2004,
    2006,
    2007,
    2008,
    2009,
    2010,
    2011,
    2013,
    2017,
    2018,
    2019
]
years_valid_split4_max_num_events_in_train_avg_valid = [
    2000,
    2002,
    2005,
    2014,
    2015,
    2016,
]
    

print("Split 4: years_train_split4_max_num_events_in_train_avg_valid")
print("Training:")
statistics_of_split(years_train_split4_max_num_events_in_train_avg_valid)
print("Validation:")
statistics_of_split(years_valid_split4_max_num_events_in_train_avg_valid)


# split5 : train_distant_years_valid_between
years_train_split5_train_distant_years_valid_between = [
    2000,
    2001,
    2002,
    2003,
    2004,
    2005,
    2006,
    2015,
    2016,
    2017,
    2018,
    2019
]
years_valid_split5_train_distant_years_valid_between = [
    2008,
    2009,
    2010,
    2011,
    2012,
    2013,
]


print("Split 5: train_distant_years_valid_between")
print("Training:")
statistics_of_split(years_train_split5_train_distant_years_valid_between)
print("Validation:")
statistics_of_split(years_valid_split5_train_distant_years_valid_between)


data_folder_split1 = data_folder+"split1_ordered/"
data_folder_split2 = data_folder+"split2_extreme_ratios_in_train_avg_in_valid/"
data_folder_split3 = data_folder+"split3_extreme_ratios_in_valid_avg_in_train/"
data_folder_split4 = data_folder+"split4_max_num_events_in_train_avg_valid/"
data_folder_split5 = data_folder+"split5_train_distant_years_valid_between/"


dataset_handler.split_train_valid(years_train_split1_ordered,years_valid_split1_ordered)
tensors_dict = dataset_handler.create_datasets(folder_path=data_folder_split1)


dataset_handler.split_train_valid(years_train_split2_extreme_ratios_in_train_avg_in_valid,
                                  years_valid_split2_extreme_ratios_in_train_avg_in_valid)
tensors_dict = dataset_handler.create_datasets(folder_path=data_folder_split2)


dataset_handler.split_train_valid(years_train_split3_extreme_ratios_in_valid_avg_in_train,
                                  years_valid_split3_extreme_ratios_in_valid_avg_in_train)
tensors_dict = dataset_handler.create_datasets(folder_path=data_folder_split3)


dataset_handler.split_train_valid(years_train_split4_max_num_events_in_train_avg_valid,
                                  years_valid_split4_max_num_events_in_train_avg_valid)
tensors_dict = dataset_handler.create_datasets(folder_path=data_folder_split4)


dataset_handler.split_train_valid(years_train_split5_train_distant_years_valid_between,
                                  years_valid_split5_train_distant_years_valid_between)
tensors_dict = dataset_handler.create_datasets(folder_path=data_folder_split5)





metadata_folder = "../../data/metadata_meteo20000101to20210630_dust_0_m24_24_48_72/"


torch.save(resulting_statistics,metadata_folder+"yearly_statistics.pkl")
torch.save(years_train_split1_ordered,metadata_folder+"years_train_split1_ordered.pkl")
torch.save(years_valid_split1_ordered,metadata_folder+"years_valid_split1_ordered.pkl")
torch.save(years_train_split2_extreme_ratios_in_train_avg_in_valid,metadata_folder+"years_train_split2_extreme_ratios_in_train_avg_in_valid.pkl")
torch.save(years_valid_split2_extreme_ratios_in_train_avg_in_valid,metadata_folder+"years_valid_split2_extreme_ratios_in_train_avg_in_valid.pkl")
torch.save(years_train_split3_extreme_ratios_in_valid_avg_in_train,metadata_folder+"years_train_split3_extreme_ratios_in_valid_avg_in_train.pkl")
torch.save(years_valid_split3_extreme_ratios_in_valid_avg_in_train,metadata_folder+"years_valid_split3_extreme_ratios_in_valid_avg_in_train.pkl")
torch.save(years_train_split4_max_num_events_in_train_avg_valid,metadata_folder+"years_train_split4_max_num_events_in_train_avg_valid.pkl")
torch.save(years_valid_split4_max_num_events_in_train_avg_valid,metadata_folder+"years_valid_split4_max_num_events_in_train_avg_valid.pkl")
torch.save(years_train_split5_train_distant_years_valid_between,metadata_folder+"years_train_split5_train_distant_years_valid_between.pkl")
torch.save(years_valid_split5_train_distant_years_valid_between,metadata_folder+"years_valid_split5_train_distant_years_valid_between.pkl")


def save_metadata(df, name):
    torch.save(df.index, name+"_times.pkl")
    torch.save(df.columns, name+"_columns.pkl")


metadata_name_all_combined = metadata_folder+"all"
metadata_name_split1 = metadata_folder+"split1_ordered"
metadata_name_split2 = metadata_folder+"split2_extreme_ratios_in_train_avg_in_valid"
metadata_name_split3 = metadata_folder+"split3_extreme_ratios_in_valid_avg_in_train"
metadata_name_split4 = metadata_folder+"split4_max_num_events_in_train_avg_valid"
metadata_name_split5 = metadata_folder+"split5_train_distant_years_valid_between"


dataset_handler.split_train_valid(years_train_split1_ordered,years_valid_split1_ordered)
save_metadata(dataset_handler.train_df,metadata_name_split1+"_train")
save_metadata(dataset_handler.valid_df,metadata_name_split1+"_valid")
dataset_handler.split_train_valid(years_train_split2_extreme_ratios_in_train_avg_in_valid,
                                  years_valid_split2_extreme_ratios_in_train_avg_in_valid)
save_metadata(dataset_handler.train_df,metadata_name_split2+"_train")
save_metadata(dataset_handler.valid_df,metadata_name_split2+"_valid")
dataset_handler.split_train_valid(years_train_split3_extreme_ratios_in_valid_avg_in_train,
                                  years_valid_split3_extreme_ratios_in_valid_avg_in_train)
save_metadata(dataset_handler.train_df,metadata_name_split3+"_train")
save_metadata(dataset_handler.valid_df,metadata_name_split3+"_valid")
dataset_handler.split_train_valid(years_train_split4_max_num_events_in_train_avg_valid,
                                  years_valid_split4_max_num_events_in_train_avg_valid)
save_metadata(dataset_handler.train_df,metadata_name_split4+"_train")
save_metadata(dataset_handler.valid_df,metadata_name_split4+"_valid")
dataset_handler.split_train_valid(years_train_split5_train_distant_years_valid_between,
                                  years_valid_split5_train_distant_years_valid_between)
save_metadata(dataset_handler.train_df,metadata_name_split5+"_train")
save_metadata(dataset_handler.valid_df,metadata_name_split5+"_valid")


save_metadata(dataset_handler.dataframe, metadata_name_all_combined)




