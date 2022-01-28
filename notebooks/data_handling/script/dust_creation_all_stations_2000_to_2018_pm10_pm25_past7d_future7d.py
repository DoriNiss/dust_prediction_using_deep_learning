#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from DustToPandasHandler_MultiStations import *
from joblib import Parallel, delayed 
import numpy as np


data_dir = "../../data"
dust_pm10_filename = f"{data_dir}/data_pm10_all_stations.csv"
dust_pm25_filename = f"{data_dir}/data_pm25_all_stations.csv"
result_filename = f"{data_dir}/dust_multistations/dust_df_all_stations_2000_to_2018_pm10_pm25_past7d_future7d"
metadata_base_filename = f"{data_dir}/dust_multistations/metadata/dust_df_all_stations_2000_to_2018_pm10_pm25_past7d_future7d_metadata"
debug_result_filename = f"{data_dir}/dust_multistations/debug_dataframe.pkl"


# lags is in hours!
lags = [0]+[i for i in range(-7*24,0,3)]+[i for i in range(3,7*24+3,3)]
print(lags, len(lags))


loaded_files = [
    DustToPandasHandler_MultiStations.load_csv_file_without_handler(dust_pm10_filename),
    DustToPandasHandler_MultiStations.load_csv_file_without_handler(dust_pm25_filename),
]


years=list(range(2000,2020))

Parallel(n_jobs=1,verbose=100)(delayed(DustToPandasHandler_MultiStations)(
        num_hours_to_avg="3h", lags=lags, delta_hours=3, saveto=f"{result_filename}_{year}.pkl", 
        avg_th=0, debug=False, keep_na=False, verbose=0, 
        stations_num_values_th=0, station_metadata_cols=None, pm_types=["PM10","PM25"], 
        metadata_base_filename=metadata_base_filename, csv_filenames=None,
        years=[year], loaded_files=loaded_files
    )    
    for year in years) 


# for year in tqdm(years):
#     print(f"\n\n####### YEARS: {year} #######")
#     dust_handler = DustToPandasHandler_MultiStations(
#         num_hours_to_avg="3h", lags=lags, delta_hours=3, saveto=f"{result_filename}_{year}.pkl", 
#         avg_th=0, debug=False, keep_na=False, verbose=0, 
#         stations_num_values_th=0, station_metadata_cols=None, pm_types=["PM10","PM25"], 
#         metadata_base_filename=metadata_base_filename, csv_filenames=None,
#         years=[year], loaded_files=loaded_files
#     )


# debug_dataframe = dust_handler.combined_dataframe#[20000:30000]
# debug_dataframe


# torch.save(debug_dataframe,debug_result_filename)


years=list(range(2000,2019))
dataframes = []
for year in years:
    try:
        df = torch.load(f"{result_filename}_{year}.pkl")
        dataframes.append(df)
        print(f"Loaded year {year}, length = {len(df)}, num_cols = {len(df.columns)}")
    except:
        continue
full_dataframe = pd.concat(dataframes)
full_dataframe.fillna(-888)
print(len(full_dataframe))


# Saving full df fails: serializing a string larger than 4 GiB requires pickle protocol 4 or higher
# SAVE THE DATAFRAME IN PARTS, JOIN ONLY THE RESULTING TENSORS
# TBD: CREATE 2019,2020


# df_2006_to_2010 = full_dataframe.loc["20060101":"20091231"]
# len(df_2006_to_2010),df_2006_to_2010.index[-1]


for year in range(2000,2019):
    yearly_df = full_dataframe.loc[f"{year}0101":f"{year}1231"]
    print(yearly_df.index[0],yearly_df.index[-1],len(yearly_df))
    torch.save(yearly_df,f"{result_filename}_full_cols_{year}.pkl")


# Supposed to be 55520
2928*5+2920*14








f"{result_filename}_full_cols_{year}.pkl"


len(full_dataframe.columns)


133*3*(14*8+1)













