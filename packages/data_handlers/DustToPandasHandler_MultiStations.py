# -*- coding: utf-8 -*-
# +
import pandas as pd
import numpy as np
import csv
import pytz
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DustToPandasHandler_MultiStations import *
from data_handlers.DustToPandasHandler_Super import *

# -

class DustToPandasHandler_MultiStations(DustToPandasHandler_Super):
    '''
        This class is used for creating a single dataframes from multiple, specific csv files
        Each csv file has multiple stations, each for different value
    '''
    def __init__(self, timezone="Asia/Jerusalem", num_hours_to_avg="3h",lags=[0,-24,24,48,72],
                 delta_hours=3, saveto=None, avg_th=3, origin_start="2000-01-01 00:00:00+00:00",
                 debug=False, keep_na=False, verbose=1,
                 stations_num_values_th=200000, station_metadata_cols=None, pm_types=["PM10","PM25"], 
                 stations_date_col_name="date",stations_hour_col_name="Hour",stations_name_col_name="Name",
                 metadata_base_filename=None, csv_filenames=[], invalid_values_flags=None,
                 only_from_stations_by_names=None, add_value_counts=True): 
        '''
            Inherits from DustToPandasHandler_Super
            Used for creating one dataframe from multiple csv files, each contains one type of dust (e.g. PM10),
            with multiple stations.
            stations_num_values_th - the minimal number of values per station needed to keep a station. 
            station_metadata_cols - cols to be saved in a metadata file (metadata_filename). 
            pm_types - list of pm types, has to be equal length with csv_filenames (corresponds to cols' names)
            csv_filenames - list of csv files to load ("new" format from Alex). Each loaded file will
                be translated into a single pandas dataframe, and then joined
            stations_hour_col_name,_date_,_name_ - according to the format of given csv files
            invalid_values_flags - special values to be filled instead of NA values in different stages,
                in this order (will be replaced by succeeding values): {
                "missing_lag": {"description":"in case na arised after calculating lags, including 0","flag":-999},
                "missing_timestamp": {"description":"in case na arised after combining stations for a single pm type","flag":-888},
                "missing_station": {"description":"in case na arised after combining pm_types","flag":-777}
            }
            get_only_from_stations_by_names - take only data from specific stations (to be tested)
            add_value_counts - add a column of values counted during averaging, per lag ["{...}_values_count_{lag}"]
            keep_na - if False, remove rows that have only NA
        '''
        super().__init__(timezone=timezone, num_hours_to_avg=num_hours_to_avg, lags=lags,
                         delta_hours=delta_hours, saveto=saveto, avg_th=avg_th, origin_start=origin_start,
                         debug=debug, keep_na=keep_na, verbose=verbose)
        self.stations_num_values_th = stations_num_values_th
        self.station_metadata_cols = station_metadata_cols or ["Name","Code","X_ITM","Y_ITM"]
        self.pm_types = pm_types
        self.stations_date_col_name = stations_date_col_name
        self.stations_hour_col_name = stations_hour_col_name
        self.stations_name_col_name = stations_name_col_name
        self.metadata_base_filename = metadata_base_filename
        self.csv_filenames = csv_filenames
        self.invalid_values_flags = invalid_values_flags or {
            "missing_lag": {"description":"in case na arised after calculating lags, including 0","flag":-999},
            "missing_timestamp": {"description":"in case na arised after combining stations for a single pm type","flag":-888},
            "missing_station": {"description":"in case na arised after combining pm_types","flag":-777}
        }
        self.only_from_stations_by_names = only_from_stations_by_names
        self.add_value_counts = add_value_counts
        print("Loading .csv data: ...")
        csv_files = [self.load_csv_file(f) for f in tqdm(csv_filenames)]
        print("... Done! Creating Pandas DataFrames (times shifted to UTC) from files: ...")
        dataframes_per_file = [self.build_single_dataframe_for_all_stations_of_one_file(csv_file,pm_type) 
                               for csv_file,pm_type in tqdm(zip(csv_files,pm_types))]
        print("... Done! Details of dataframes per pm type: ...")
        for i,pm_type in enumerate(pm_types):
            self.print_dataframe(dataframes_per_file[i])
        print("Joining dataframes of all pm types: ...")
        combined_dataframe = dataframes_per_file[0]
        if len(dataframes_per_file)>1:
            for df in dataframes_per_file[1:]:
                combined_dataframe = combined_dataframe.join(df, how="outer")
        value_to_fill = self.invalid_values_flags["missing_station"]["flag"]
        combined_dataframe = combined_dataframe.fillna(value_to_fill)
        self.combined_dataframe = combined_dataframe
        print(f"... Done! Missing stations filled with {value_to_fill}")
        print(f"Result:")
        self.print_dataframe(combined_dataframe)
        if self.saveto is not None:
            self.save_dataframe(combined_dataframe,saveto)
                              
    def build_single_dataframe_for_all_stations_of_one_file(self,dust_csv,pm_type):
        dust_dataframes_per_station = self.build_dataframes_list_per_station(dust_csv,pm_type)
        print(f"Calculating {self.num_hours_to_avg} averages and lags for each station: ...")
        dust_averaged_dataframes_per_station = []
        for station_df in tqdm(dust_dataframes_per_station):
            station_averaged_df = self.calculate_averages_for_dataframe(station_df)
            station_averaged_df = self.get_single_station_df_with_lags(station_averaged_df)
            if not self.keep_na:
                station_averaged_df = self.drop_na_if_all_lags_are_na(station_averaged_df)
#                 station_averaged_df = station_averaged_df.dropna(how="all")
            value_to_fill = self.invalid_values_flags["missing_lag"]["flag"]
            station_averaged_df = station_averaged_df.fillna(value_to_fill)
            dust_averaged_dataframes_per_station.append(station_averaged_df)
        print(f"... Done! Missing lags filled with {value_to_fill}. Combining stations' dataframes: ...")
        combined_dataframe = dust_averaged_dataframes_per_station[0]
        if len(dust_averaged_dataframes_per_station)>1:
            for df in dust_averaged_dataframes_per_station[1:]:
                combined_dataframe = combined_dataframe.join(df, how="outer")
        value_to_fill = self.invalid_values_flags["missing_timestamp"]["flag"]
        combined_dataframe = combined_dataframe.fillna(value_to_fill)
        print(f"... Done! Missing timestamps filled with {value_to_fill}")
        return combined_dataframe
            
    def build_dataframes_list_per_station(self,dust_csv,pm_type):
        """ dust_csv is the full list of csv rows, with the titles row at dust_csv[0]"""
        print("... Building metadata for each station: ...")   
        stations_metadata = self.get_stations_metadata_from_full_csv(dust_csv,pm_type)
        print(f"... Done, got {len(stations_metadata)} stations. Separating csv information per station: ...")   
        dust_csv_per_station = []
        for station_metadata in stations_metadata:
            first_idx,last_idx = station_metadata["first_csv_idx"],station_metadata["last_csv_idx"]
            if last_idx-first_idx<self.stations_num_values_th: 
                continue
            if self.only_from_stations_by_names is not None:
                if station_metadata[self.stations_name_col_name] not in self.only_from_stations_by_names:
                    continue
            dust_csv_per_station.append([dust_csv[0]]+dust_csv[first_idx:last_idx])
        dust_dataframes_per_station = []
        print(f"... Done! Creating DataFrames per station: ...")   
        for dust_csv_station in tqdm(dust_csv_per_station):
            try: 
                dust_df = self.build_dataframe_from_station_csv(dust_csv_station,pm_type)
            except Exception as e: 
                if self.verbose>0: print(e)
                continue
            if dust_df is not None and not dust_df.empty:
                dust_dataframes_per_station.append(dust_df)
        print(f"... Done! Resulted with {len(dust_dataframes_per_station)} stations")   
        return dust_dataframes_per_station            

    def get_stations_metadata_from_full_csv(self,dust_csv,pm_type="test"):
        """
            Returns a list of dicts with the following information per station:
                "first_csv_idx": first idx of the station in the csv file
                "last_csv_idx": last idx of the station in the csv file (excluded)
                i.e.: dust_csv[first_idx:last_idx] are the rows of that station
            Assuming the first row contains titles and there are more than 2 ros of data. 
            Event's threshold will be added later by a different method with a different class 
            pm_type - suffix to be added to metadata_filename for saving
        """
        stations_metadata = []
        col_to_check = self.station_metadata_cols[0]
        stations_metadata_idxs = {}
        for col in self.station_metadata_cols:
            stations_metadata_idxs[col] = dust_csv[0].index(col)
        stations_metadata.append({col: dust_csv[1][stations_metadata_idxs[col]] 
                                  for col in self.station_metadata_cols})
        stations_metadata[0]["first_csv_idx"]=1
        for row_idx_minus_2,row in enumerate(dust_csv[2:]):
            current_metadata = {col: row[stations_metadata_idxs[col]] for col in self.station_metadata_cols}
            if current_metadata[col_to_check]!=stations_metadata[-1][col_to_check]:
                stations_metadata[-1]["last_csv_idx"]=row_idx_minus_2+2
                stations_metadata.append(current_metadata)
                stations_metadata[-1]["first_csv_idx"]=row_idx_minus_2+2
            if row==dust_csv[-1]:
                stations_metadata[-1]["last_csv_idx"]=row_idx_minus_2+3
        if self.metadata_base_filename is not None:
            torch.save(stations_metadata+[{"invalid_values_flags":self.invalid_values_flags}],
                       f"{self.metadata_base_filename}_{pm_type}.pkl")
        return stations_metadata

    def build_dataframe_from_station_csv(self,station_dust_csv,pm_type="PM10"):
        """ Input: list of csv rows, with the titles row. Output: pd.DataFrame """  
        dust_col_idx = station_dust_csv[0].index(pm_type)
        date_col_idx = station_dust_csv[0].index(self.stations_date_col_name)
        hour_col_idx = station_dust_csv[0].index(self.stations_hour_col_name)
        name_col_idx = station_dust_csv[0].index(self.stations_name_col_name)
        station_name = station_dust_csv[1][name_col_idx]
        print(f"... ... Building DataFrame for station {station_name}: ...")   
        rows_dust,timestamps = [],[]
        for row in station_dust_csv[1:]:
            try: 
                h = float(row[hour_col_idx])
                hours_str = f"{row[hour_col_idx]}:00:00" if h-int(h)==0 else f"{row[hour_col_idx]}:30:00"
                timestamp_str = f"{row[date_col_idx]} {hours_str}"
                shifted_utc_timestamp = pd.to_datetime(timestamp_str).tz_localize(self.timezone).tz_convert('UTC')
            except Exception as e: 
                if self.verbose>0: print("... ... Warning: an error occured while translating a timestamp, "\
                                    "ignoring row: ",e,row)
                continue
            try:
                row_dust=float(row[dust_col_idx])
            except Exception as e: 
                if row[dust_col_idx]!="NA" and verbose>0: print("... ... Warning: an error occured while " \
                                                                "translating a dust value, ignoring row:",e,row)
                continue
            if row_dust<=0:
                continue
            rows_dust.append(row_dust)
            timestamps.append(shifted_utc_timestamp)
        if self.verbose>0: print(f"... ... Done! Result has {len(rows_dust)} rows")   
        if len(rows_dust)<self.stations_num_values_th: 
            if self.verbose>0: print(f"... ... Ignoring station: less than {self.stations_num_values_th} values")   
            return
        return pd.DataFrame({f"{station_name}_{pm_type}_0":rows_dust},index=timestamps) 
    
    def calculate_averages_for_dataframe(self,dust_dataframe):
        dust_grouped = dust_dataframe.groupby(pd.Grouper( 
            freq=self.num_hours_to_avg, origin=self.origin_start,label="left"))
        idxs = dust_grouped.count()>=self.avg_th
        col_name = dust_dataframe.columns[0]
        idxs = idxs[col_name].values
        dust_avgs = dust_grouped.mean()
        if self.add_value_counts:
            counts = dust_grouped.count()[dust_grouped.count().columns[0]].values
            counts_name = f"{col_name[:-2]}_values_count_0"
            dust_avgs[counts_name] = counts
        dust_avgs = dust_avgs[idxs]
        return dust_avgs

    def get_single_station_df_with_lags(self, station_df):
        """ Assuming the dust column name is of shape <STATION>_<PM_TYPE>_0. lags are in hours""" 
        base_name = station_df.columns[0][:-2]
        for lag in self.lags: # splitted into loops so all lags and all deltas are together
            shift_name = f"{base_name}_{lag}" if lag>=0 else f"{base_name}_m{-lag}"
            dusts_lag = station_df[f"{base_name}_0"].shift(periods=-lag,freq="h")
            station_df[shift_name] = dusts_lag
        for lag in self.lags:
            delta_name = f"{base_name}_delta_{lag}" if lag>=0 else f"{base_name}_delta_m{-lag}"
            dusts_lag = station_df[f"{base_name}_0"].shift(periods=-lag,freq="h")
            dusts_just_before_lag = dusts_lag.shift(periods=self.delta_hours,freq="h")
            station_df[delta_name] = dusts_lag-dusts_just_before_lag
        if self.add_value_counts:
            for lag in self.lags:
                counts_name = f"{base_name}_values_count_{lag}" if lag>=0 else f"{base_name}_values_count_m{-lag}"
                counts_lag = station_df[f"{base_name}_values_count_0"].shift(periods=-lag,freq="h")
                station_df[counts_name] = counts_lag
        return station_df
    
    def drop_na_if_all_lags_are_na(self, df):
        def row_is_na(i,cols):
            for col in cols:
                if not np.isnan(df[i:i+1][col].values[0]):
                    return False     
            return True
        lags_cols = []
        clean_df = df
        for c in df.columns:
            if "values_count" not in c: lags_cols.append(c)
        for i in range(1,len(df)):
            if row_is_na(i,lags_cols):
                clean_df = clean_df.drop(df.index[i])
        return clean_df



    
