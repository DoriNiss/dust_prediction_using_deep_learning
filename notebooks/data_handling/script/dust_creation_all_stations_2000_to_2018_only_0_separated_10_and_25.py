#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from DustToPandasHandler_MultiStations import *
import numpy as np


data_dir = "../../data"
dust_pm10_filename = f"{data_dir}/data_pm10_all_stations.csv"
dust_pm25_filename = f"{data_dir}/data_pm25_all_stations.csv"
result_filename_10 = f"{data_dir}/dust_multistations/dust_df_0_all_stations_200000v_2000to2018_pm10.pkl"
result_filename_25 = f"{data_dir}/dust_multistations/dust_df_0_all_stations_200000v_2000to2018_pm25.pkl"
metadata_base_filename_10 = f"{data_dir}/dust_multistations/metadata/dust_df_0_all_stations_200000v_2000to2018_pm10_metadata"
metadata_base_filename_25 = f"{data_dir}/dust_multistations/metadata/dust_df_0_all_stations_200000v_2000to2018_pm25_metadata"


# lags is in hours!
lags = [0]
print(lags)


dust_handler_10 = DustToPandasHandler_MultiStations(
    num_hours_to_avg="3h", lags=lags, delta_hours=3, saveto=result_filename_10, 
    avg_th=3, debug=False, keep_na=False, verbose=0, 
    stations_num_values_th=200000, station_metadata_cols=None, pm_types=["PM10"], 
    metadata_base_filename=metadata_base_filename_10, csv_filenames=[dust_pm10_filename], 
    na_value_combined=-99999, only_from_stations_by_names=None
)


dust_handler_25 = DustToPandasHandler_MultiStations(
    num_hours_to_avg="3h", lags=lags, delta_hours=3, saveto=result_filename_25, 
    avg_th=3, debug=False, keep_na=False, verbose=0, 
    stations_num_values_th=200000, station_metadata_cols=None, pm_types=["PM25"], 
    metadata_base_filename=metadata_base_filename_25, csv_filenames=[dust_pm25_filename], 
    na_value_combined=-99999, only_from_stations_by_names=None
)














idxs = np.arange(10000,10010)
lag = 72
idxs_diff = -lag//3
idxs_lag = idxs+idxs_diff
lag_name = "dust_"+str(lag)
print(dust_handler.dust_lags.index[idxs])
print(dust_handler.dust_lags.index[idxs_lag])
print(dust_handler.dust_lags["dust_0"][idxs])
print(dust_handler.dust_lags[lag_name][idxs_lag])




