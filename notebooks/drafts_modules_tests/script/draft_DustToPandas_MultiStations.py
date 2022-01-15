#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import csv
from tqdm import tqdm

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DustToPandasHandler_MultiStations import *


data_dir = "../../data"
dust_pm10_filename = f"{data_dir}/data_pm10_all_stations.csv"
dust_pm25_filename = f"{data_dir}/data_pm25_all_stations.csv"
debug_filename_10 = f"{data_dir}/data_pm10_all_stations_debug.csv"
debug_filename_25 = f"{data_dir}/data_pm25_all_stations_debug.csv"

destination_folder = f"{data_dir}/dust_multistations/debug"
debug_metadata_base_filename = f"{destination_folder}/metadata_debug"
debug_result_filename = f"{destination_folder}/debug_dataframe.pkl"


handler = DustToPandasHandler_MultiStations(
    num_hours_to_avg="3h", lags=[0,-24,24,48,72], delta_hours=3, saveto=debug_result_filename, 
    avg_th=3, debug=False, keep_na=False, verbose=1, 
    stations_num_values_th=4000, station_metadata_cols=None, pm_types=["PM10","PM25"], 
    metadata_base_filename=destination_folder, csv_filenames=[debug_filename_10,debug_filename_25], 
    na_value_combined=-99999, only_from_stations_by_names=None
)













