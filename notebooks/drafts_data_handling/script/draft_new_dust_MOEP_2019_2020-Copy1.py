#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from DustToPandasHandler_MultiStations import *
from joblib import Parallel, delayed 
import numpy as np


data_dir = "../../data"
dust_dir = f"{data_dir}/dust_MOEP_2019_2020"
dust_filename = f"{dust_dir}/dust_MOEP_raw_2019to2020_pm10_pm25.csv"
dust_filename_debug = f"{dust_dir}/dust_MOEP_raw_2019to2020_pm10_pm25_debug.csv"

dust_pm10_older_filename = f"{data_dir}/data_pm10_all_stations.csv"
dust_df_metadata_filename = f"{dust_dir}/metadata/dust_df_all_stations_2000_to_2018_pm10_pm25_past7d_future7d_metadata_all.pkl"

# dust_pm25_filename = f"{data_dir}/data_pm25_all_stations.csv"
# result_filename = f"{data_dir}/dust_multistations/dust_df_all_stations_2000_to_2018_pm10_pm25_past7d_future7d"
# metadata_base_filename = f"{data_dir}/dust_multistations/metadata/dust_df_all_stations_2000_to_2018_pm10_pm25_past7d_future7d_metadata"
# debug_result_filename = f"{data_dir}/dust_multistations/debug_dataframe.pkl"


def load_csv_file(filename):
    file = open(filename, 'r') 
    return list(csv.reader(file))


dust_debug_csv = load_csv_file(dust_filename_debug)


len(dust_debug_csv[0]),len(dust_debug_csv[1])


dust_debug_csv[0]


dust_older = load_csv_file(dust_pm10_older_filename)


dust_older[0],dust_older[1]


dust_df_metadata = torch.load(dust_df_metadata_filename)


len(dust_df_metadata["PM10"]),len(dust_df_metadata["PM25"])


dust_df_metadata


for row_idx in range(len(dust_debug_csv[0])):
    print(dust_debug_csv[0][row_idx],dust_debug_csv[1][row_idx])


for s in dust_df_metadata["PM10"]:
    if 'Name' not in s.keys():
        continue
    print(f"{s['Name']}, x: {s['X_ITM']}, y: {s['Y_ITM']}, PM10")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
for s in dust_df_metadata["PM25"]:
    if 'Name' not in s.keys():
        continue
    print(f"{s['Name']}, x: {s['X_ITM']}, y: {s['Y_ITM']}, PM2.5")







