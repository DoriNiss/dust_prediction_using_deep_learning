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
dust_filename_debug = f"{dust_dir}/dust_MOEP_raw_2019to2020_pm10_pm25_debug.csv"
base_stations_filename = f"{dust_dir}/base_stations.pkl"


dust_pm10_older_filename = f"{data_dir}/data_pm10_all_stations.csv"
dust_df_metadata_filename = f"{dust_dir}/metadata/dust_df_all_stations_2000_to_2018_pm10_pm25_past7d_future7d_metadata_all.pkl"

# dust_pm25_filename = f"{data_dir}/data_pm25_all_stations.csv"
# result_filename = f"{data_dir}/dust_multistations/dust_df_all_stations_2000_to_2018_pm10_pm25_past7d_future7d"
# metadata_base_filename = f"{data_dir}/dust_multistations/metadata/dust_df_all_stations_2000_to_2018_pm10_pm25_past7d_future7d_metadata"
# debug_result_filename = f"{data_dir}/dust_multistations/debug_dataframe.pkl"





base_stations_filename = f"{dust_dir}/base_stations.pkl"
base_stations_list = torch.load(base_stations_filename)


handler = DustToPandasHandler_MultiStations_MOEP_hebrew(
    timezone="Asia/Jerusalem", num_hours_to_avg="3h",lags=[0,-24,24,48,72],
    delta_hours=3, saveto=None, avg_th=0, origin_start="2019-01-01 00:00:00+00:00",
    debug=False, keep_na=False, verbose=1,
    data_csv_filename=dust_filename_debug, loaded_file=None, stations_num_values_th=0, 
    pm_csv_titles_to_base_titles_translation={"PM10":"PM10","PM2.5":"PM25"},
    base_stations=base_stations_list, hebrew_station_titles_row=0, pm_types_titles_row=1, invalid_values_flags=None,
    only_from_stations_by_names=None, add_value_counts=True, years=None,invalid_rows_idxs=[2,3,4],
    timestamp_format_day_first=True,
)


base_filename = f"{dust_dir}/debug"
handler.create_and_save_batched_dataframes(base_filename, batch_size=None, num_jobs=1)






































df = handler.create_dataframe_from_rows(0,17)
df


df["MAVKIIM_PM25_values_count_48"]


num_rows = 49
batch_size = 10
for num_batch in range(num_rows//batch_size):
    first_row,last_row = num_batch*batch_size,(num_batch+1)*batch_size
    print(first_row,last_row)
if last_row != num_rows:
    first_row,last_row = last_row,num_rows
    print(first_row,last_row)


def calc_first_and_last_row_from_batch_num(batch_size=None, num_batch=None):
    num_rows = 50 #len(self.csv_file)
    if batch_size is None:
        first_row,last_row = 0,num_rows
    else:
        first_row,last_row = num_batch*batch_size,min((num_batch+1)*batch_size,num_rows)
    if first_row >= num_rows:
        return None,None
#     if first_row >= num_rows//batch_size or last_row >= num_rows:
#         first_row,last_row = batch_size*num_rows//batch_size,num_rows
    return first_row,last_row


for i in range(7):
    print(calc_first_and_last_row_from_batch_num(10,i))





invalid_values_mask = df.isna() #df==invalid_lag_value or 
lags_cols = [col for col in df if "values_count" not in col]
values_counts_cols = [col for col in df if "values_count" in col]
df_na = df[invalid_values_mask][lags_cols]
df_na = df_na.fillna(-999)
df_na
df[invalid_values_mask][lags_cols] = df_na
df[invalid_values_mask][lags_cols]
# df[invalid_values_mask][values_counts_cols] = 0














df[lags_cols] = np.where(df[lags_cols].isna(),-999,df[lags_cols])
df[values_counts_cols] = np.where(df[values_counts_cols].isna(),0,df[values_counts_cols])
df


df["AMIEL_PM25_24"],df["AMIEL_PM25_values_count_24"]














mask1 = df[[col for col in df.columns if "values_count" not in col]]==-999

















np.nan


handler.calculate_averages_for_dataframe(df)

















ts1 = pd.to_datetime("2000-01-01 00:00")
ts1


ts1+2*pd.Timedelta("30min")==pd.to_datetime("2000-01-01 01:00")


[(row_idx,row[0]) for row_idx,row in enumerate(handler.csv_file)]

















# handler.csv_row_to_timestamp(52)





df1 = pd.DataFrame({"A":[1,2,3],"B":[10,20,30]},
    index=[handler.csv_row_to_timestamp(6),handler.csv_row_to_timestamp(7),handler.csv_row_to_timestamp(8)])
df1


df1["C"] = [100,200,300]
df1


df1[["B","C"]]











t2,t1+2*pd.Timedelta("30min")


handler.csv_row_to_timestamp(3)


handler.csv_row_to_timestamp(1) is pd.NaT











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







