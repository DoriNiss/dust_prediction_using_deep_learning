#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import csv
from tqdm import tqdm

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DustToPandasHandler import *


data_dir = "../../data"
dust_pm10_filename = f"{data_dir}/data_pm10_all_stations.csv"
dust_pm25_filename = f"{data_dir}/data_pm25_all_stations.csv"
debug_filename = f"{data_dir}/data_pm10_all_stations_debug.csv"
debug_filename_25 = f"{data_dir}/data_pm25_all_stations_debug.csv"


def load_csv_dust_file_new_format(filename):
    file = open(filename, 'r') 
    return list(csv.reader(file))


for filename in tqdm([dust_pm10_filename]):
    dust_raw_10 = load_csv_dust_file_new_format(filename)


print(dust_raw_10[:3],dust_raw_10[-4:-1])


for filename in tqdm([dust_pm25_filename]):
    dust_raw_25 = load_csv_dust_file_new_format(filename)


print(dust_raw_25[:3],dust_raw_25[-4:-1])


print(len(dust_raw_10),len(dust_raw_25))


dust_raw_10[0]


debug_10 = [dust_raw_10[0]]+dust_raw_10[91000:96000]+dust_raw_10[3090000:3095000]
print(len(debug_10),debug_10[:3],debug_10[-4:-1])


dust_raw_25[0]


debug_25 = [dust_raw_25[0]]+dust_raw_25[91000:96000]+dust_raw_25[4600000:4605000]
print(len(debug_25),debug_25[:3],debug_25[-4:-1])


# [d[8] for d in dust_raw_25[4600000:4605000]]


with open(debug_filename, mode='w') as f:
#     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer = csv.writer(f)
    for row in debug_10:
        csv_writer.writerow(row)


with open(debug_filename_25, mode='w') as f:
#     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer = csv.writer(f)
    for row in debug_25:
        csv_writer.writerow(row)


for filename in tqdm([debug_filename]):
    debug_10 = load_csv_dust_file_new_format(filename)
print(len(debug_10),debug_10[:3],debug_10[-4:-1])


"""
Plan:
    DustToPandas:
    
    Create DF per station
    Remove NA and <= 0
    Create description file of that station (name,short_name,x,y,...)
    Average (3h)
    Create lags
    Combine all dataframes
    Fill NA with a predefined number (default -9999)
    
    Dataset:
    Load the new DF
    Create targets tensor - 3D: [timestamps,stations,dust_values]
    Create regional targets - 3D [timestamps,[North,Center,South],dust_values]
    Add to the metadata descriptor: {stations: {number_of_station_channel,name,short_name,x,y...}}
"""


import pandas as pd
import numpy as np
import torch
import csv
from tqdm import tqdm

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DustToPandasHandler import *

data_dir = "../../data"
dust_pm10_filename = f"{data_dir}/data_pm10_all_stations.csv"
dust_pm25_filename = f"{data_dir}/data_pm25_all_stations.csv"
debug_filename = f"{data_dir}/data_pm10_all_stations_debug.csv"

def load_csv_dust_file_new_format(filename):
    file = open(filename, 'r') 
    return list(csv.reader(file))

for filename in tqdm([debug_filename]):
    debug_10 = load_csv_dust_file_new_format(filename)
print(len(debug_10),debug_10[:3],debug_10[-4:-1])


stations_num_values_th = 4000
stations_dust_col_name = "PM10"
stations_date_col_name = "date"
stations_hour_col_name = "Hour"
stations_name_col_name = "Name"
timezone="Asia/Jerusalem"
num_hours_to_avg="3h"
lags=[0,-24,24,48,72]
delta_hours=3
avg_th=3
origin_start="2000-01-01 00:00:00+00:00"
na_value = -99999





def get_stations_metadata_from_full_csv(dust_csv,save_metadata_as=None):
    """
        Returns a list of dicts with the following information per station:
            station_metadata_cols (default: ["Name","Code","X_ITM","Y_ITM"])
            "first_idx": first idx of the station in the csv file
            "last_idx": last idx of the station in the csv file (excluded)
            i.e.: dust_csv[first_idx:last_idx] are the rows of that station
        Assuming the first row contains titles and there are more than 2 ros of data. 
        Event's threshold will be added later by a different method with a different class 
    """
    stations_metadata = []
    station_metadata_cols = ["Name","Code","X_ITM","Y_ITM"] 
    col_to_check = station_metadata_cols[0]
    stations_metadata_idxs = {}
    for col in station_metadata_cols:
        stations_metadata_idxs[col] = dust_csv[0].index(col)
    stations_metadata.append({col: dust_csv[1][stations_metadata_idxs[col]] for col in station_metadata_cols})
    stations_metadata[0]["first_csv_idx"]=1
    for row_idx_minus_2,row in enumerate(dust_csv[2:]):
        current_metadata = {col: row[stations_metadata_idxs[col]] for col in station_metadata_cols}
        if current_metadata[col_to_check]!=stations_metadata[-1][col_to_check]:
            stations_metadata[-1]["last_csv_idx"]=row_idx_minus_2+2
            stations_metadata.append(current_metadata)
            stations_metadata[-1]["first_csv_idx"]=row_idx_minus_2+2
        if row==dust_csv[-1]:
            stations_metadata[-1]["last_csv_idx"]=row_idx_minus_2+3
    if save_metadata_as is not None:
        torch.save(stations_metadata,save_metadata_as)
    return stations_metadata

def calculate_averages_for_dataframe(dust_dataframe):
    dust_grouped = dust_dataframe.groupby(pd.Grouper( 
        freq=num_hours_to_avg, origin=origin_start,label="left"))
    idxs = dust_grouped.count()>=avg_th
    col_name = dust_dataframe.columns[0]
    idxs = idxs[col_name].values
    dust_avgs = dust_grouped.mean()[idxs]
    return dust_avgs

def get_single_station_df_with_lags(station_df):
    """ Assuming the dust column name is of shape <STATION>_<PM_TYPE>_0. lags are in hours""" 
    base_name = station_df.columns[0][:-2]
    for lag in lags: # splitted into 2 loops so all lags and all deltas are together
        shift_name = f"{base_name}_{lag}" if lag>=0 else f"{base_name}_m{-lag}"
        dusts_lag = station_df[f"{base_name}_0"].shift(periods=-lag,freq="h")
        station_df[shift_name] = dusts_lag
    for lag in lags:
        delta_name = f"{base_name}_delta_{lag}" if lag>=0 else f"{base_name}_delta_m{-lag}"
        dusts_lag = station_df[f"{base_name}_0"].shift(periods=-lag,freq="h")
        dusts_just_before_lag = dusts_lag.shift(periods=delta_hours,freq="h")
        station_df[delta_name] = dusts_lag-dusts_just_before_lag
    return station_df

def build_one_dataframe_for_all_stations(dust_csv):
    dust_dataframes_per_station = build_dataframes_list_per_station(dust_csv)
    print(f"## Calculating {num_hours_to_avg} averages and lags for each station...")
    dust_averaged_dataframes_per_station = []
    for station_df in tqdm(dust_dataframes_per_station):
        station_averaged_df = calculate_averages_for_dataframe(station_df)
        station_averaged_df = get_single_station_df_with_lags(station_averaged_df)
        dust_averaged_dataframes_per_station.append(station_averaged_df)
    print(f"## ...Done! Combining dataframes...")
    combined_dataframe = dust_averaged_dataframes_per_station[0]
    if len(dust_averaged_dataframes_per_station)>1:
        for df in dust_averaged_dataframes_per_station[1:]:
            combined_dataframe = combined_dataframe.join(df, how="outer")
    print(f"## ...Done!")
    combined_dataframe = combined_dataframe.fillna(na_value)
    return combined_dataframe

def build_dataframes_list_per_station(dust_csv):
    """ dust_csv is the full list of csv rows, with the titles row at dust_csv[0]"""
    print("## Building metadata for each station...")   
    stations_metadata = get_stations_metadata_from_full_csv(dust_csv)
    dust_csv_per_station = []
    print(f"## ...Done, got {len(stations_metadata)} stations. Separating csv information per station...")   
    for station_metadata in stations_metadata:
        first_idx,last_idx = station_metadata["first_csv_idx"],station_metadata["last_csv_idx"]
        if last_idx-first_idx<stations_num_values_th: 
            continue
        dust_csv_per_station.append([dust_csv[0]]+dust_csv[first_idx:last_idx])
    dust_dataframes_per_station = []
    print(f"## ...Done! Creating DataFrames per station...")   
    for dust_csv_station in tqdm(dust_csv_per_station):
        try: 
            dust_df = build_dataframe_from_station_csv(dust_csv_station)
        except Exception as e: 
            print(e)
            continue
        if not dust_df.empty:
            dust_dataframes_per_station.append(dust_df)
    print(f"## ...Done! Resulted with {len(dust_dataframes_per_station)} stations")   
    return dust_dataframes_per_station

def build_dataframe_from_station_csv(station_dust_csv,cols_title_prefix="PM10",verbose=0):
    """ Input: list of csv rows, with the titles row. Output: pd.DataFrame """  
    dust_col_idx = station_dust_csv[0].index(stations_dust_col_name)
    date_col_idx = station_dust_csv[0].index(stations_date_col_name)
    hour_col_idx = station_dust_csv[0].index(stations_hour_col_name)
    name_col_idx = station_dust_csv[0].index(stations_name_col_name)
    station_name = station_dust_csv[1][name_col_idx]
    print(f"### ...Creating for station {station_name}...")   
    rows_dust,timestamps = [],[]
    for row in station_dust_csv[1:]:
        try: 
            h = float(row[hour_col_idx])
            hours_str = f"{row[hour_col_idx]}:00:00" if h-int(h)==0 else f"{row[hour_col_idx]}:30:00"
            timestamp_str = f"{row[date_col_idx]} {hours_str}"
            shifted_utc_timestamp = pd.to_datetime(timestamp_str).tz_localize(timezone).tz_convert('UTC')
        except Exception as e: 
            if verbose>0: print("### Warning: an error occured while translating a timestamp, ignoring row:",                                 e,row)
            continue
        try:
            row_dust=float(row[dust_col_idx])
        except Exception as e: 
            if row[dust_col_idx]!="NA" and verbose>0: print("### Warning: an error occured while "                                                             "translating a dust value, ignoring row:",e,row)
            continue
        if row_dust<=0:
            continue
        rows_dust.append(row_dust)
        timestamps.append(shifted_utc_timestamp)
    print(f"### ...Done! Result has {len(rows_dust)} rows")   
    if len(rows_dust)<stations_num_values_th: 
        print(f"### Ignoring station")   
        return
    return pd.DataFrame({f"{station_name}_{cols_title_prefix}_0":rows_dust},index=timestamps)


debug_full_df = build_one_dataframe_for_all_stations(debug_10)
print(debug_full_df)
print(debug_full_df[500:600])





# df.fillna(0)
# df.replace(np.nan, 0)


# print(debug_full_df[500:530])


dataframes[0].columns[0]


get_stations_metadata_from_full_csv(debug_10)





a = np.nan
float("NA")


df_test = pd.DataFrame({"dust_0":[]},index=[]) 


df_test.empty


debug_10[78:83]

















### ADD COUNTS


num_hours_to_avg,origin_start,avg_th


debug_full_df_raw = build_dataframe_from_station_csv(debug_10,cols_title_prefix="PM10",verbose=0)
debug_full_df_raw


def calculate_averages_for_dataframe(dust_dataframe):
    dust_grouped = dust_dataframe.groupby(pd.Grouper( 
        freq=num_hours_to_avg, origin=origin_start,label="left"))
    idxs = dust_grouped.count()>=avg_th
#         counts = ...
    # check how it looks like, how to add column of counts
    col_name = dust_dataframe.columns[0]
    idxs = idxs[col_name].values
    dust_avgs = dust_grouped.mean()
    counts = dust_grouped.count()[dust_grouped.count().columns[0]].values
    counts_name = f"{col_name[:-2]}_values_count_0"
    dust_avgs[counts_name] = counts
    dust_avgs = dust_avgs[idxs]
    return dust_avgs


calculate_averages_for_dataframe(debug_full_df_raw)


test_df = calculate_averages_for_dataframe(debug_full_df_raw)
test_df["test"] = test_df["AFULA_PM10_0"].shift(periods=-24,freq="h")
test_df


np.isnan(test_df["test"][-1])


def drop_na_if_all_lags_are_na(df):
    def row_is_na(i,cols):
        for col in cols:
            if not np.isnan(df[i:i+1][col].values[0]):
                return False     
        return True
    lags_cols = []
    clean_df = df
    for c in df.columns:
        if "values_count" not in c and "AFULA_PM10_0" not in c: lags_cols.append(c)
    for i in range(1,len(df)):
        if row_is_na(i,lags_cols):
            clean_df = clean_df.drop(df.index[i])
    return clean_df


drop_na_if_all_lags_are_na(test_df)








test_df


"""
    DATASET (TENSORS)
"""


debug_full_df


"""
    Dataset:
    Load the new DF
    Create targets tensor - 4D: [timestamps,stations,dust_type,dust_lags_and_deltas]
    Create regional targets - 4D [timestamps,[North,Center,South],dust_type,dust_lags_and_deltas]
    Add to the metadata descriptor: {stations: {number_of_station_channel,name,short_name,x,y...}}
"""


metadata_df_10 = get_stations_metadata_from_full_csv(debug_10)


num_stations = len(metadata_df_10)
timestamps = debug_full_df.index
N = len(debug_full_df.index)
targets_tensor = torch.zeros([N,num_stations,1,len(lags)*2])+na_value
targets_tensor.shape


a=["A_PM10_0","A_PM10_24","A_PM25_0","A_PM25_24",
   "B_PM10_0","B_PM10_24","B_PM25_0","B_PM25_24"]
a


[col for col in a if "PM10" in col]





""" CREATE METADATA """


""" 
    Info to keep:
    stations and their metadata, pm types, lags_and_deltas_description {index: description}
    timestamps - out of the metadata dict
"""


pm_types = ["PM10","PM25"]


""" Assuming all pm types dataframes have the same lags"""

def get_lags_and_deltas_idxs_dict_from_df(df,sample_station="AFULA"):
    def lag_str_to_description(lag_str):
        if lag_str=="0":
            description = "T"
        elif lag_str[0]=="m":
            description = f"T-{lag_str[1:]}h"
        else:
            description = f"T+{lag_str}h"
        return description
    col_strings = df.columns[[sample_station in col for col in df.columns]]
    col_strings_lags = col_strings[["delta" not in col for col in col_strings]]
    col_strings_deltas = col_strings[["delta" in col for col in col_strings]]
    num_lags = len(col_strings_lags)
    lags_and_deltas_idxs_dict = {i: {} for i in range(len(col_strings))}
    for i,lag_str_full in enumerate(col_strings_lags):
        lag_str = lag_str_full[lag_str_full.rindex("_")+1:len(lag_str_full)]
        lag_description = lag_str_to_description(lag_str_full[lag_str_full.rindex("_")+1:len(lag_str_full)])
        lags_and_deltas_idxs_dict[i]["description"] = f"Dust at {lag_description}"
        lags_and_deltas_idxs_dict[i]["col_lag_suffix"] = lag_str
    for i,lag_str_full in enumerate(col_strings_lags):
        lag_str = lag_str_full[lag_str_full.rindex("_")+1:len(lag_str_full)]
        lag_description = lag_str_to_description(lag_str)
        lags_and_deltas_idxs_dict[num_lags+i]["description"] = f"Delta at {lag_description}"
        lags_and_deltas_idxs_dict[num_lags+i]["col_lag_suffix"] = f"delta_{lag_str}"
    return lags_and_deltas_idxs_dict


get_lags_and_deltas_idxs_dict_from_df(debug_full_df)


s = 'AFULA_PM10_0'
s[s.rindex("_")+1:len(s)]


metadata_df_10


metadata_name_key = "Name"

targets_metadata = {
    "stations_metadata": metadata_df_10,
    "targets_shape": "[timestamps,stations,pm_types,lags_and_deltas+1]", # [-1 - value counts]
    "stations_idxs": {i: station[metadata_name_key] for i,station in enumerate(metadata_df_10)},
    "pm_types_idxs": {i: pm_type for i,pm_type in enumerate(pm_types)},
    "lags_and_deltas_idxs": get_lags_and_deltas_idxs_dict_from_df(debug_full_df,sample_station=metadata_df_10[0][metadata_name_key]),
}
targets_metadata


a = [23,24,25]
{n: i for i,n in enumerate(a)}


""" CREATE TENSORS """


num_pm_types = len(targets_metadata["pm_types_idxs"])
num_lags = len(targets_metadata["lags_and_deltas_idxs"])

num_stations = len(metadata_df_10)
timestamps = debug_full_df.index
N = len(debug_full_df.index)
targets_tensor = torch.zeros([N,num_stations,num_pm_types,num_lags])+na_value
targets_tensor.shape


pm_types_dict = targets_metadata["pm_types_idxs"]
stations_dict = targets_metadata["stations_idxs"]
lags_and_deltas_dict = targets_metadata["lags_and_deltas_idxs"]

# for row in tqdm(range(N)):
#     for pm_type in pm_types_dict.keys():
#         for station in stations_dict.keys():
#             for lag_and_delta in lags_and_deltas_dict.keys():
#                 lag_suffix = lags_and_deltas_dict[lag_and_delta]["col_lag_suffix"]
#                 df_col = f"{stations_dict[station]}_{pm_types_dict[pm_type]}_{lag_suffix}"
#                 if df_col in debug_full_df.columns:
#                     print(debug_full_df[row:row+1][df_col].values)
#                     targets_tensor[row,pm_type,station,lag_and_delta] = debug_full_df[row:row+1][df_col]

print("Populating targets tensor...")
for pm_type in tqdm(pm_types_dict.keys()):
    for station in stations_dict.keys():
        for lag_and_delta in lags_and_deltas_dict.keys():
            lag_suffix = lags_and_deltas_dict[lag_and_delta]["col_lag_suffix"]
            df_col = f"{stations_dict[station]}_{pm_types_dict[pm_type]}_{lag_suffix}"
            if df_col in debug_full_df.columns:
                targets_tensor[:,pm_type,station,lag_and_delta] = torch.tensor(debug_full_df[:][df_col].values)
print(f"...Done! Shape: {targets_tensor.shape}")


targets_tensor[targets_tensor==-99999].shape,targets_tensor.view(-1).shape













