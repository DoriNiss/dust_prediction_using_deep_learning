# -*- coding: utf-8 -*-
# +
import pandas as pd
import numpy as np
import csv
import pytz
import torch
from tqdm import tqdm
from joblib import Parallel, delayed #conda install -c anaconda joblib

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DustToPandasHandler_MultiStations import *
from data_handlers.DustToPandasHandler_Super import *
from data_handlers.MOEP_hebrew_station_tranlations import *




# -

class DustToPandasHandler_MultiStations_MOEP_hebrew(DustToPandasHandler_MultiStations):
    '''
        This class is used for creating a single dataframes from multiple, specific csv files
        Each csv file has multiple stations, each for different value
    '''
    def __init__(self, timezone="Asia/Jerusalem", num_hours_to_avg="3h",lags=[0,-24,24,48,72],
                 delta_hours=3, saveto=None, avg_th=0, origin_start="2019-01-01 00:00:00+00:00",
                 debug=False, keep_na=False, verbose=1,
                 data_csv_filename=None, loaded_file=None, stations_num_values_th=0, 
                 pm_csv_titles_to_base_titles_translation={"PM10":"PM10","PM2.5":"PM25"},
                 base_stations=None, hebrew_station_titles_row=0, pm_types_titles_row=1, invalid_values_flags=None,
                 only_from_stations_by_names=None, add_value_counts=True, years=None, freq="30min", invalid_rows_idxs=[2],
                 timestamp_format_day_first=True, 
            ): 
        '''
            Inherits from DustToPandasHandler_MultiStations
            Used for creating one dataframe from a single csv file downloaded from https://www.svivaaqm.net/ 
            data_csv_filename - csv file to load
            loaded_file - if not None, don't load the data as it is already given
            stations_num_values_th - the minimal number of values per station needed to keep a station (TODO). 
            pm_csv_titles_to_base_titles_translation - dict of used pm types corresponding to df cols' names
            hebrew_station_titles_row,pm_types_titles_row - idx of according rows from the csv file
            base_stations - list of stations' names from another source to be joined later as a tensor 
                (e.g. df from another format)
            invalid_values_flags - special values to be filled instead of NA values in different stages,
                in this order (will be replaced by succeeding values): {
                "missing_lag": {"description":"in case na arised after calculating lags or value, including 0","flag":-999},
                "missing_station": {"description":"in case na arised after combining base_stations","flag":-777}
            }
            only_from_stations_by_names - take only data from specific stations (defaults to base_station)
            add_value_counts - add a column of values counted during averaging, per lag ["{...}_values_count_{lag}"]
            keep_na - if False, remove rows that have only NA
            years - if not None, create dataframes only for certain years (useful for when data is too large)
            invalid_rows_idxs - e.g. known empty rows
        '''
        DustToPandasHandler_Super.__init__(self, timezone=timezone, num_hours_to_avg=num_hours_to_avg, lags=lags,
                                           delta_hours=delta_hours, saveto=saveto, avg_th=avg_th, origin_start=origin_start,
                                           debug=debug, keep_na=keep_na, verbose=verbose) #TODO: use super()
        self.data_csv_filename = data_csv_filename
        self.stations_num_values_th = stations_num_values_th
        self.pm_csv_titles_to_base_titles_translation = pm_csv_titles_to_base_titles_translation
        self.base_stations = base_stations
        self.hebrew_station_titles_row = hebrew_station_titles_row
        self.pm_types_titles_row = pm_types_titles_row
        self.invalid_values_flags = invalid_values_flags
        self.only_from_stations_by_names = only_from_stations_by_names or base_stations
        self.add_value_counts = add_value_counts
        self.invalid_rows_idxs = invalid_rows_idxs
        self.freq = freq
        self.years = years
        self.timestamp_format_day_first = timestamp_format_day_first
        self.invalid_values_flags = invalid_values_flags or {
            "missing_lag": {"description":"in case na arised after calculating lags or values, including 0","flag":-999},
            "missing_station": {"description":"in case na arised after combining base_stations","flag":-777}
        }
        if loaded_file is None:
            print("Loading .csv data: ...")
            csv_file = self.load_csv_file(data_csv_filename)
        else:
            csv_file = loaded_file
        self.csv_file = csv_file
        self.cols_dict = {}
        self.init_cols_dict()
        print(f"Initiated handler! Number of rows in csv file: {len(csv_file)}, number of cols: {len(self.cols_dict)}")
        print(f"Use self.create_and_save_batched_dataframes(base_filename,batch_size=None,num_jobs=2) to create " \
              f"and save dataframes (batch_size=None means one batch for all file, so no parallelism)")
        
    def init_cols_dict(self):
        cols_dict = {}
        csv_stations_titles = self.csv_file[self.hebrew_station_titles_row]
        csv_pm_types_titles = self.csv_file[self.pm_types_titles_row]
        if len(csv_stations_titles) != len(csv_pm_types_titles):
            raise Exception(f"Error! Stations and PM types rows have to be of the same size, but got " \
                            f"stations len {len(csv_stations_titles)} and pm types len {len(csv_pm_types_titles)}. Aborting...")
        len_rows_csv = len(csv_stations_titles)
        for i in range(1,len_rows_csv): # first is title
            hebrew_name = csv_stations_titles[i] 
            if hebrew_name=="" and i>1:
                if i-1 not in cols_dict.keys(): continue
                hebrew_name=cols_dict[i-1]["hebrew_name"] # Assuming blank title always follows a valid one of the same place
            station_name = get_pm10_pm25_translated_station_name_from_hebrew(hebrew_name)
            pm_csv_title = csv_pm_types_titles[i]
            if pm_csv_title not in self.pm_csv_titles_to_base_titles_translation.keys() or station_name is None: 
                continue
            pm_type = self.pm_csv_titles_to_base_titles_translation[pm_csv_title]
            full_name = f"{station_name}_{pm_type}"
            cols_dict[i] = {"full_name":full_name,"station_name":station_name,":pm_type":pm_type,"hebrew_name":hebrew_name}
        self.cols_dict = cols_dict
        
    def create_and_save_batched_dataframes(self, base_filename, batch_size=None, num_jobs=2): # None means all file
        if batch_size is None:
            self.create_and_save_dataframe_from_num_batch_and_size(base_filename, batch_size=batch_size, num_batch=None)
            return
        num_batches = len(self.csv_file)//batch_size+1 # exists if batch number is too large
        Parallel(n_jobs=num_jobs,verbose=100)(delayed(self.create_and_save_dataframe_from_num_batch_and_size)(
            base_filename, batch_size=batch_size, num_batch=num_batch) for num_batch in range(num_batches))
        # Creates overlaps!
    
    def create_and_save_dataframe_from_num_batch_and_size(self, base_filename, batch_size=None, num_batch=None):
        def calc_first_and_last_row_from_batch_num(batch_size, num_batch):
            num_rows = len(self.csv_file)
            if batch_size is None:
                first_row,last_row = 0,num_rows
            else:
                first_row,last_row = num_batch*batch_size,min((num_batch+1)*batch_size,num_rows)
            if first_row >= num_rows:
                return None,None
            return first_row,last_row
        first_row,last_row = calc_first_and_last_row_from_batch_num(batch_size, num_batch)
        if first_row is None or last_row is None: return
        self.create_and_save_dataframe_from_rows(base_filename, first_row, last_row, num_batch=num_batch)
        
    def create_and_save_dataframe_from_rows(self, base_filename, first_row, last_row, num_batch=None): #not including last row
        df = self.create_dataframe_from_rows(first_row, last_row)
        filename = f"{base_filename}.pkl" if num_batch is None else f"{base_filename}_b{num_batch}.pkl"
        torch.save(df,filename)
        print(f"Saved dataframe of batch #{num_batch}, of len {len(df)}")
    
    def create_dataframe_from_rows(self, first_row, last_row):
        data_as_dict = {f"{station_dict['full_name']}_0":[] for station_dict in self.cols_dict.values()}
        timestamps = []
        # Create one dataframe for all columns in rows range
        for row_idx in tqdm(range(first_row,last_row)):
            row_as_dict = {}
            if row_idx==self.hebrew_station_titles_row or row_idx==self.pm_types_titles_row: continue
            row = self.csv_file[row_idx]
            timestamp = self.csv_row_to_timestamp(row_idx)
            if timestamp is None: continue
            if self.years is not None:
                if timestamp.year not in self.years:
                    continue
            timestamps.append(timestamp)
            invalid_lag_value = np.nan 
            for sation_col,station_dict in self.cols_dict.items():
                try:
                    station_value = float(row[sation_col])
                except:
                    station_value = invalid_lag_value
                if station_value<=0:
                    station_value = invalid_lag_value
                data_as_dict[f"{station_dict['full_name']}_0"].append(station_value)
        dataframe = pd.DataFrame(data_as_dict,index=timestamps)                 
        # Calculate averages, deltas, values counts
        dataframe_averaged = self.calculate_averages_for_dataframe(dataframe)
        # Add missing stations from base_stations to dataframe and keep only only_from_stations_by_names
        dataframe_averaged = self.add_base_stations_and_keep_only_wanted(dataframe_averaged)
        # Create lags
        dataframe_averaged = self.calculate_lags(dataframe_averaged)
        # Fill na with flags and their values_count with 0
        dataframe_averaged = self.fill_invalid_lag_value_and_values_counts(dataframe_averaged)
        # Drop rows that don't have valid lags at all
        dataframe_averaged = self.drop_rows_without_any_lag(dataframe_averaged)
        return dataframe_averaged

    def csv_row_to_timestamp(self, row_idx):
        try:
            timestamp_csv_str = self.csv_file[row_idx][0]
            timestamp = pd.to_datetime(timestamp_csv_str,dayfirst=self.timestamp_format_day_first)
            if timestamp is pd.NaT:
                raise Exception;
            try:
                timestamp = timestamp.tz_localize(self.timezone).tz_convert('UTC')
                return timestamp
            except:
                return None
        except Exception:
            if row_idx==0 or row_idx in self.invalid_rows_idxs:
                return None
            prev_timestamp = self.csv_row_to_timestamp(row_idx-1)
            if prev_timestamp is None:
                return None
            if row_idx==len(self.csv_file)-1:
                return prev_timestamp+pd.Timedelta(self.freq)
            next_timestamp = self.csv_row_to_timestamp(row_idx+1)
            if next_timestamp==prev_timestamp+2*pd.Timedelta(self.freq):
                return prev_timestamp+pd.Timedelta(self.freq)
           
    def calculate_averages_for_dataframe(self,dust_dataframe):
        dust_grouped = dust_dataframe.groupby(pd.Grouper( 
            freq=self.num_hours_to_avg, origin=self.origin_start,label="left"))
        where_more_than_th_counts = dust_grouped.count()>=self.avg_th
        dust_avgs = dust_grouped.mean()
        counts_as_dict = {}
        for col_name in dust_dataframe.columns:
            if self.add_value_counts:
                counts = dust_grouped.count()[col_name].values
                counts_name = f"{col_name[:-2]}_values_count_0"
                counts_as_dict[counts_name] = counts
            idxs_col_more_than_th = where_more_than_th_counts[col_name].values
            dust_avgs[col_name][~idxs_col_more_than_th] = np.nan
        if self.add_value_counts:
            dust_avgs = pd.concat([dust_avgs,pd.DataFrame(counts_as_dict,index=dust_avgs.index)],axis=1)
        return dust_avgs

    def add_base_stations_and_keep_only_wanted(self, df): 
        base_stations_as_dict = {}
        if self.base_stations is not None:
            no_station_values = [self.invalid_values_flags["missing_station"]["flag"]]*len(df)
            for base_sation in self.base_stations:
                for pm_str in self.pm_csv_titles_to_base_titles_translation.values():
                    base_station_full_name = f"{base_sation}_{pm_str}_0"
                    if base_station_full_name in df.columns: continue
                    base_stations_as_dict[base_station_full_name] = no_station_values
                    if self.add_value_counts:
                        base_stations_as_dict[f"{base_sation}_{pm_str}_values_count_0"] = [0]*len(df)
            df = pd.concat([df,pd.DataFrame(base_stations_as_dict,index=df.index)],axis=1)
        if self.only_from_stations_by_names is not None:
            cols_to_keep = []
            for station_to_keep in self.only_from_stations_by_names:
                for pm_str in self.pm_csv_titles_to_base_titles_translation.values():
                    cols_to_keep.append(f"{station_to_keep}_{pm_str}_0")
                    if self.add_value_counts: cols_to_keep.append(f"{station_to_keep}_{pm_str}_values_count_0")
            df = df[cols_to_keep]
        return df

    def calculate_lags(self, df):
        """ Assuming the dust column name is of shape <STATION>_<PM_TYPE>_0. lags are in hours""" 
        cols_to_calc = [col_name for col_name in df.columns if "values_count" not in col_name]
        lags_as_dict = {}
        for col_name in cols_to_calc:
            base_name = col_name[:-2]
            for lag in self.lags: 
                if lag==0: continue
                shift_name = f"{base_name}_{lag}" if lag>0 else f"{base_name}_m{-lag}"
                dusts_lag = df[col_name].shift(periods=-lag,freq="h")
                lags_as_dict[shift_name] = dusts_lag
            for lag in self.lags:
                delta_name = f"{base_name}_delta_{lag}" if lag>=0 else f"{base_name}_delta_m{-lag}"
                dusts_lag = df[f"{base_name}_0"].shift(periods=-lag,freq="h")
                dusts_just_before_lag = dusts_lag.shift(periods=self.delta_hours,freq="h")
                lags_as_dict[delta_name] = dusts_lag-dusts_just_before_lag
            if self.add_value_counts:
                for lag in self.lags:
                    if lag==0: continue
                    counts_name = f"{base_name}_values_count_{lag}" if lag>0 else f"{base_name}_values_count_m{-lag}"
                    counts_lag = df[f"{base_name}_values_count_0"].shift(periods=-lag,freq="h")
                    lags_as_dict[counts_name] = counts_lag
        df = df.join(pd.DataFrame(lags_as_dict,index=df.index),how="inner")
        return df
    
    def fill_invalid_lag_value_and_values_counts(self, df):
        lags_cols = [col for col in df.columns if "values_count" not in col]
        values_counts_cols = [col for col in df.columns if "values_count" in col]
        invalid_lag_value = self.invalid_values_flags["missing_lag"]["flag"]
        df[lags_cols] = np.where(df[lags_cols].isna(),invalid_lag_value,df[lags_cols])
        df[values_counts_cols] = np.where(df[values_counts_cols].isna(),0,df[values_counts_cols])
        return df
   
    def drop_rows_without_any_lag(self, df):
        invalid_flags = [flag_dict["flag"] for flag_dict in self.invalid_values_flags.values()]
        def row_is_na(i,cols):
            for col in cols:
                v = df[i:i+1][col].values[0]
                if not np.isnan(v) and v not in invalid_flags:
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
       
        
    
