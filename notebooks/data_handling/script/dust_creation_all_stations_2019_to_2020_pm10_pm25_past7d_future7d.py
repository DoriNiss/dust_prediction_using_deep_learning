#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from DustToPandasHandler_MultiStations_MOEP_hebrew import *
from joblib import Parallel, delayed 
import numpy as np


data_dir = "../../data"
dust_dir = f"{data_dir}/dust_MOEP_2019_2020"
dust_filename = f"{dust_dir}/dust_MOEP_raw_2019to2020_pm10_pm25.csv"
base_stations_filename = f"{dust_dir}/base_stations.pkl"

result_filename = f"{dust_dir}/dataframes/dust_df_MOEP_2019_2020"

# dust_pm10_older_filename = f"{data_dir}/data_pm10_all_stations.csv"
# dust_df_metadata_filename = f"{dust_dir}/metadata/dust_df_all_stations_2000_to_2018_pm10_pm25_past7d_future7d_metadata_all.pkl"


base_stations_list = torch.load(base_stations_filename)

lags = [0]+[i for i in range(-7*24,0,3)]+[i for i in range(3,7*24+3,3)]
print(lags, len(lags))


handler = DustToPandasHandler_MultiStations_MOEP_hebrew(
    timezone="Asia/Jerusalem", num_hours_to_avg="3h",lags=lags,
    delta_hours=3, saveto=None, avg_th=0, origin_start="2019-01-01 00:00:00+00:00",
    debug=False, keep_na=False, verbose=1,
    data_csv_filename=dust_filename, loaded_file=None, stations_num_values_th=0, 
    pm_csv_titles_to_base_titles_translation={"PM10":"PM10","PM2.5":"PM25"},
    base_stations=base_stations_list, hebrew_station_titles_row=2, pm_types_titles_row=3, 
    invalid_values_flags=None,
    only_from_stations_by_names=None, add_value_counts=True, years=None, invalid_rows_idxs=[0,1,4],
    timestamp_format_day_first=True,
)


handler.create_and_save_batched_dataframes(result_filename, batch_size=17544, num_jobs=1)





73224/339/2


result1 = torch.load(f"{result_filename}_b0.pkl")
result1[1:]


result2 = torch.load(f"{result_filename}_b1.pkl")
result2


df_all = pd.concat([result1[1:],result2])
df_all


result_filename


torch.save(df_all,f"{result_filename}_full.pkl")







