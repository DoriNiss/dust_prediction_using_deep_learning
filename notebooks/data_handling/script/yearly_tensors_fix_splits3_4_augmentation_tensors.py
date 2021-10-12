#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from Dataset_handler import *
import numpy as np
import pandas as pd
import torch


# This notebook fixes splits 3 and 4, creates yearly splits with their times, and creates augmentation 
# tensors for each of the 5 splits


data_dir = "../../data/tensors_meteo20000101to20210630_dust_0_m24_24_48_72/"
split1_dir = data_dir+"split1_ordered/"
split2_dir = data_dir+"split2_extreme_ratios_in_train_avg_in_valid/"
split3_dir = data_dir+"split3_extreme_ratios_in_valid_avg_in_train/"
split4_dir = data_dir+"split4_max_num_events_in_train_avg_valid/"
split5_dir = data_dir+"split5_train_distant_years_valid_between/"
split_dirs = [split1_dir,split2_dir,split3_dir,split4_dir,split5_dir]
meteorology_train_paths = [split+"tensor_train_meteorology.pkl" for split in split_dirs]
meteorology_valid_paths = [split+"tensor_valid_meteorology.pkl" for split in split_dirs]
dust_train_paths = [split+"tensor_train_dust.pkl" for split in split_dirs]
dust_valid_paths = [split+"tensor_valid_dust.pkl" for split in split_dirs]

metadata_dir = "../../data/metadata_meteo20000101to20210630_dust_0_m24_24_48_72/"
metadata_columns_path = metadata_dir+"all_columns.pkl"
metadata_yearly_statistics_path = metadata_dir+"yearly_statistics.pkl"
metadata_all_times_path = metadata_dir+"all_times.pkl"
metadata_times_split1_train_path = metadata_dir+"split1_ordered_train_times.pkl"
metadata_times_split2_train_path = metadata_dir+"split2_extreme_ratios_in_train_avg_in_valid_train_times.pkl"
metadata_times_split3_train_path = metadata_dir+"split3_extreme_ratios_in_valid_avg_in_train_train_times.pkl"
metadata_times_split4_train_path = metadata_dir+"split4_max_num_events_in_train_avg_valid_train_times.pkl"
metadata_times_split5_train_path = metadata_dir+"split5_train_distant_years_valid_between_train_times.pkl"
metadata_times_train_paths = [metadata_times_split1_train_path,metadata_times_split2_train_path,
                              metadata_times_split3_train_path,metadata_times_split4_train_path,
                              metadata_times_split5_train_path]
metadata_times_split1_valid_path = metadata_dir+"split1_ordered_valid_times.pkl"
metadata_times_split2_valid_path = metadata_dir+"split2_extreme_ratios_in_train_avg_in_valid_valid_times.pkl"
metadata_times_split3_valid_path = metadata_dir+"split3_extreme_ratios_in_valid_avg_in_train_valid_times.pkl"
metadata_times_split4_valid_path = metadata_dir+"split4_max_num_events_in_train_avg_valid_valid_times.pkl"
metadata_times_split5_valid_path = metadata_dir+"split5_train_distant_years_valid_between_valid_times.pkl"
metadata_times_valid_paths = [metadata_times_split1_valid_path,metadata_times_split2_valid_path,
                              metadata_times_split3_valid_path,metadata_times_split4_valid_path,
                              metadata_times_split5_valid_path]


# split1 : ordered
years_train_split1_ordered = [y for y in range(2000,2012)]
years_valid_split1_ordered = [y for y in range(2013,2019)]


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


print("1:", "train:", years_train_split1_ordered, "validation:", years_valid_split1_ordered)
print("2:", "train:", years_train_split2_extreme_ratios_in_train_avg_in_valid, "validation:", years_valid_split2_extreme_ratios_in_train_avg_in_valid)
print("1:", "train:", years_train_split5_train_distant_years_valid_between, "validation:", years_valid_split5_train_distant_years_valid_between)


# We will take all the years from split 1 (train+valid), and add from split 2 train the years 2012 and 2019


times_split_1_train = torch.load(metadata_times_split1_train_path)
len(times_split_1_train)


meteorology_split_1_train = torch.load(meteorology_train_paths[0])
meteorology_split_1_train.shape


dust_split_1_train = torch.load(dust_train_paths[0])
dust_split_1_train.shape


times_split_1_valid = torch.load(metadata_times_split1_valid_path)
meteorology_split_1_valid = torch.load(meteorology_valid_paths[0])
print(len(times_split_1_valid), meteorology_split_1_valid.shape)


dust_split_1_valid = torch.load(dust_valid_paths[0])
print(dust_split_1_valid.shape)


times_split_2_train = torch.load(metadata_times_split2_train_path)
meteorology_split_2_train = torch.load(meteorology_train_paths[1])
print(len(times_split_2_train), meteorology_split_2_train.shape) 


dust_split_2_train = torch.load(dust_train_paths[1])
print(dust_split_2_train.shape) 


path_to_yearly_data = data_dir+"yearly_data/"
path_to_yearly_data


mask_idxs = times_split_1_train[:].year==2000
meteorology_split_1_train[mask_idxs].shape, mask_idxs, len(mask_idxs)


all_years = [y for y in range(2000,2020)]
num_datapoints_total = 0
yearly_data_dict = {
    "years": [],
    "meteorology": [],
    "dust": [],
    "times": []
}
for year in all_years:
    print("Year:", year)
    if year == 2012 or year == 2019:
        meteorology_big = meteorology_split_2_train
        dust_big = dust_split_2_train
        times_big = times_split_2_train
    elif year<=2011:
        meteorology_big = meteorology_split_1_train
        dust_big = dust_split_1_train
        times_big = times_split_1_train
    else:
        meteorology_big = meteorology_split_1_valid
        dust_big = dust_split_1_valid
        times_big = times_split_1_valid
    print("      Sizes full:", meteorology_big.shape, dust_big.shape, len(times_big))   
    mask_idxs = times_big[:].year == year
    meteorology = meteorology_big[mask_idxs]
    dust = dust_big[mask_idxs]
    times = times_big[mask_idxs]
    print("      Sizes yearly selected:", meteorology.shape, dust.shape, len(times))  
    print(times)
#     torch.save(meteorology, path_to_yearly_data+"meteorology_"+str(year)+".pkl")
#     torch.save(dust, path_to_yearly_data+"dust_"+str(year)+".pkl")
#     torch.save(times, path_to_yearly_data+"times_"+str(year)+".pkl")
    num_datapoints_total+=len(times)
    yearly_data_dict["years"].append(year)
    yearly_data_dict["meteorology"].append(meteorology)
    yearly_data_dict["dust"].append(dust)
    yearly_data_dict["times"].append(times)
print("Total datapoints:",num_datapoints_total)
    
        
        


# fix split 3 and split 4


print("3:", "train:", years_train_split3_extreme_ratios_in_valid_avg_in_train, "validation:", years_valid_split3_extreme_ratios_in_valid_avg_in_train)
print("4:", "train:", years_train_split4_max_num_events_in_train_avg_valid, "validation:", years_valid_split4_max_num_events_in_train_avg_valid)


def combine_years(years):
    data_idx = yearly_data_dict["years"].index(years[0])
    meteorology = yearly_data_dict["meteorology"][data_idx]
    dust = yearly_data_dict["dust"][data_idx]
    times = yearly_data_dict["times"][data_idx]
    for year in years[1:]:
        data_idx = yearly_data_dict["years"].index(year)
        meteorology = torch.cat((meteorology,yearly_data_dict["meteorology"][data_idx]),0)
        dust = torch.cat((dust,yearly_data_dict["dust"][data_idx]),0)
        times=times.union(yearly_data_dict["times"][data_idx])
    print(f"Combined {years}, shapes: {meteorology.shape}, {dust.shape}, {times.shape}")
    return meteorology ,dust, times 
        


# times_new = yearly_data_dict["times"][0].union(yearly_data_dict["times"][1])
# times_new


t,d,times = combine_years([2000,2001,2002])
t.shape, d.shape, times


t[0,0,0,0], t[1234,0,0,0], yearly_data_dict["meteorology"][0][0,0,0,0], yearly_data_dict["meteorology"][1][0,0,0,0]


t[-1,0,0,0], yearly_data_dict["meteorology"][2][-1,0,0,0]


# correct! Let's fix the datasets


meteorology_split3_train, dust_split3_train, times_split3_train = combine_years(years_train_split3_extreme_ratios_in_valid_avg_in_train)


meteorology_split3_valid, dust_split3_valid, times_split3_valid = combine_years(years_valid_split3_extreme_ratios_in_valid_avg_in_train)


meteorology_split4_train, dust_split4_train, times_split4_train = combine_years(years_train_split4_max_num_events_in_train_avg_valid)


meteorology_split4_valid, dust_split4_valid, times_split4_valid = combine_years(years_valid_split4_max_num_events_in_train_avg_valid)


# metadata_times_split3_train_path = metadata_dir+"split3_extreme_ratios_in_valid_avg_in_train_train_times.pkl"
# metadata_times_split4_train_path = metadata_dir+"split4_max_num_events_in_train_avg_valid_train_times.pkl"
# metadata_times_split3_valid_path = metadata_dir+"split3_extreme_ratios_in_valid_avg_in_train_valid_times.pkl"
# metadata_times_split4_valid_path = metadata_dir+"split4_max_num_events_in_train_avg_valid_valid_times.pkl"
# split3_dir = data_dir+"split3_extreme_ratios_in_valid_avg_in_train/"
# split4_dir = data_dir+"split4_max_num_events_in_train_avg_valid/"
# meteorology_train_paths = [split+"tensor_train_meteorology.pkl" for split in split_dirs]
# meteorology_valid_paths = [split+"tensor_valid_meteorology.pkl" for split in split_dirs]
# dust_train_paths = [split+"tensor_train_dust.pkl" for split in split_dirs]
# dust_valid_paths = [split+"tensor_valid_dust.pkl" for split in split_dirs]

# torch.save(times_split3_train,metadata_times_split3_train_path)
# torch.save(times_split3_valid,metadata_times_split3_valid_path)
# torch.save(times_split4_train,metadata_times_split4_train_path)
# torch.save(times_split4_valid,metadata_times_split4_valid_path)


meteorology_train_paths[2]


# not enough disk space?? fixed

# torch.save(meteorology_split3_train, meteorology_train_paths[2])
# torch.save(meteorology_split3_valid, meteorology_valid_paths[2])
# torch.save(dust_split3_train, dust_train_paths[2])
# torch.save(dust_split3_valid, dust_valid_paths[2])

# torch.save(meteorology_split4_train, meteorology_train_paths[3])
# torch.save(meteorology_split4_valid, meteorology_valid_paths[3])
# torch.save(dust_split4_train, dust_train_paths[3])
# torch.save(dust_split4_valid, dust_valid_paths[3])








# Recalling that:
# resulting_statistics = {
#     2000: {"events": 162 , "clear": 1072 , "total": 1234 , "ratio": 15.11 },
#     2001: {"events": 160 , "clear": 2090 , "total": 2250 , "ratio": 7.66  },
#     2002: {"events": 228 , "clear": 1636 , "total": 1864 , "ratio": 13.94 },
#     2003: {"events": 411 , "clear": 1634 , "total": 2045 , "ratio": 25.15 },
#     2004: {"events": 306 , "clear": 1983 , "total": 2289 , "ratio": 15.43 },
#     2005: {"events": 217 , "clear": 1838 , "total": 2055 , "ratio": 11.81 },
#     2006: {"events": 309 , "clear": 2099 , "total": 2408 , "ratio": 14.72 },
#     2007: {"events": 274 , "clear": 2367 , "total": 2641 , "ratio": 11.58 },
#     2008: {"events": 278 , "clear": 2207 , "total": 2485 , "ratio": 12.6  },
#     2009: {"events": 251 , "clear": 2044 , "total": 2295 , "ratio": 12.28 },
#     2010: {"events": 423 , "clear": 2129 , "total": 2552 , "ratio": 19.87 },
#     2011: {"events": 282 , "clear": 1349 , "total": 1631 , "ratio": 20.9  },
#     2012: {"events": 151 , "clear": 619  , "total": 770  , "ratio": 24.39 },
#     2013: {"events": 292 , "clear": 1881 , "total": 2173 , "ratio": 15.52 },
#     2014: {"events": 176 , "clear": 1722 , "total": 1898 , "ratio": 10.22 },
#     2015: {"events": 198 , "clear": 1707 , "total": 1905 , "ratio": 11.6  },
#     2016: {"events": 246 , "clear": 2055 , "total": 2301 , "ratio": 11.97 },
#     2017: {"events": 273 , "clear": 1910 , "total": 2183 , "ratio": 14.29 },
#     2018: {"events": 278 , "clear": 2074 , "total": 2352 , "ratio": 13.4  },
#     2019: {"events": 334 , "clear": 1618 , "total": 1952 , "ratio": 20.64 },
#     "total":{"events": 5249 ,"clear": 36034, "total": 41283 ,"ratio": 14.57 }
# }


# Create dataset for presentation! 18 training (cuda was out of memory for more), 1 validation, 1 augmentation
# valid year: 2017 (average ratio of events, not too long ago)
# augmentation year: 2011 (not the highest num of days but not the lowest)

presentation_years_train = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,
                            2012,2013,2014,2015,2016,2018,2019]
presentation_years_valid = [2017]
presentation_years_augmentation = [2011]

presentation_dir = data_dir+"presentation_set/"

meteorology_presentation_train, dust_presentation_train, times_presentation_train = combine_years(presentation_years_train)
meteorology_presentation_valid, dust_presentation_valid, times_presentation_valid = combine_years(presentation_years_valid)
meteorology_presentation_augmentation, dust_presentation_augmentation, times_presentation_augmentation = combine_years(presentation_years_augmentation)


torch.save(meteorology_presentation_train,presentation_dir+"tensor_train_meteorology.pkl")
torch.save(dust_presentation_train,presentation_dir+"tensor_train_dust.pkl")
torch.save(times_presentation_train,presentation_dir+"times_train.pkl")

torch.save(meteorology_presentation_valid,presentation_dir+"tensor_valid_meteorology.pkl")
torch.save(dust_presentation_valid,presentation_dir+"tensor_valid_dust.pkl")
torch.save(times_presentation_valid,presentation_dir+"times_valid.pkl")

torch.save(meteorology_presentation_augmentation,presentation_dir+"tensor_augmentation_meteorology.pkl")
torch.save(dust_presentation_augmentation,presentation_dir+"tensor_augmentation_dust.pkl")
torch.save(times_presentation_augmentation,presentation_dir+"times_augmentation.pkl")








# augmentation datasets for 5 splits
print("1:", "train:", years_train_split1_ordered, "validation:", years_valid_split1_ordered)
print()
print("2:", "train:", years_train_split2_extreme_ratios_in_train_avg_in_valid, "validation:", years_valid_split2_extreme_ratios_in_train_avg_in_valid)
print()
print("3:", "train:", years_train_split3_extreme_ratios_in_valid_avg_in_train, "validation:", years_valid_split3_extreme_ratios_in_valid_avg_in_train)
print()
print("4:", "train:", years_train_split4_max_num_events_in_train_avg_valid, "validation:", years_valid_split4_max_num_events_in_train_avg_valid)
print()
print("5:", "train:", years_train_split5_train_distant_years_valid_between, "validation:", years_valid_split5_train_distant_years_valid_between)


augmentation_years_split1 = [2012, 2019]
augmentation_years_split2 = [2008, 2009]
augmentation_years_split3 = [2007, 2014]
augmentation_years_split4 = [2001, 2012]
augmentation_years_split5 = [2007, 2014]


meteorology_split1_augmentation, dust_split1_augmentation, times_split1_augmentation = combine_years(augmentation_years_split1)
meteorology_split2_augmentation, dust_split2_augmentation, times_split2_augmentation = combine_years(augmentation_years_split2)
meteorology_split3_augmentation, dust_split3_augmentation, times_split3_augmentation = combine_years(augmentation_years_split3)
meteorology_split4_augmentation, dust_split4_augmentation, times_split4_augmentation = combine_years(augmentation_years_split4)
meteorology_split5_augmentation, dust_split5_augmentation, times_split5_augmentation = combine_years(augmentation_years_split5)


torch.save(meteorology_split1_augmentation,split1_dir+"tensor_augmentation_meteorology.pkl")
torch.save(dust_split1_augmentation,split1_dir+"tensor_augmentation_dust.pkl")
torch.save(times_split1_augmentation,split1_dir+"times_augmentation.pkl")

torch.save(meteorology_split2_augmentation,split2_dir+"tensor_augmentation_meteorology.pkl")
torch.save(dust_split2_augmentation,split2_dir+"tensor_augmentation_dust.pkl")
torch.save(times_split2_augmentation,split2_dir+"times_augmentation.pkl")

torch.save(meteorology_split3_augmentation,split3_dir+"tensor_augmentation_meteorology.pkl")
torch.save(dust_split3_augmentation,split3_dir+"tensor_augmentation_dust.pkl")
torch.save(times_split3_augmentation,split3_dir+"times_augmentation.pkl")

torch.save(meteorology_split4_augmentation,split4_dir+"tensor_augmentation_meteorology.pkl")
torch.save(dust_split4_augmentation,split4_dir+"tensor_augmentation_dust.pkl")
torch.save(times_split4_augmentation,split4_dir+"times_augmentation.pkl")

torch.save(meteorology_split5_augmentation,split5_dir+"tensor_augmentation_meteorology.pkl")
torch.save(dust_split5_augmentation,split5_dir+"tensor_augmentation_dust.pkl")
torch.save(times_split5_augmentation,split5_dir+"times_augmentation.pkl")


# test sets - loading, shaped, loading from Dataset Loader


print("split1:")
print(torch.load(split1_dir+"tensor_augmentation_meteorology.pkl").shape)
print(torch.load(split1_dir+"tensor_augmentation_dust.pkl").shape)
print(torch.load(split1_dir+"times_augmentation.pkl"))

print("split2:")
print(torch.load(split2_dir+"tensor_augmentation_meteorology.pkl").shape)
print(torch.load(split2_dir+"tensor_augmentation_dust.pkl").shape)
print(torch.load(split2_dir+"times_augmentation.pkl"))

print("split3:")
print(torch.load(split3_dir+"tensor_augmentation_meteorology.pkl").shape)
print(torch.load(split3_dir+"tensor_augmentation_dust.pkl").shape)
print(torch.load(split3_dir+"times_augmentation.pkl"))

print("split4:")
print(torch.load(split4_dir+"tensor_augmentation_meteorology.pkl").shape)
print(torch.load(split4_dir+"tensor_augmentation_dust.pkl").shape)
print(torch.load(split4_dir+"times_augmentation.pkl"))

print("split5:")
print(torch.load(split5_dir+"tensor_augmentation_meteorology.pkl").shape)
print(torch.load(split5_dir+"tensor_augmentation_dust.pkl").shape)
print(torch.load(split5_dir+"times_augmentation.pkl"))


print("presentation:")

print(torch.load(presentation_dir+"tensor_train_meteorology.pkl").shape)
print(torch.load(presentation_dir+"tensor_train_dust.pkl").shape)
print(torch.load(presentation_dir+"times_train.pkl"))

print(torch.load(presentation_dir+"tensor_valid_meteorology.pkl").shape)
print(torch.load(presentation_dir+"tensor_valid_dust.pkl").shape)
print(torch.load(presentation_dir+"times_valid.pkl"))

print(torch.load(presentation_dir+"tensor_augmentation_meteorology.pkl").shape)
print(torch.load(presentation_dir+"tensor_augmentation_dust.pkl").shape)
print(torch.load(presentation_dir+"times_augmentation.pkl"))







