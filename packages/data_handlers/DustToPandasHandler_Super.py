# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv
import pytz
import torch

class DustToPandasHandler_Super:
    '''
        This class is the super class of DustToPandas handlers, which are used for taking a dust file 
        and transform it into a pandas dataframe. This df can be later used for tensor creation.
        
    '''
    def __init__(self, timezone="Asia/Jerusalem", num_hours_to_avg="3h",lags=[0,-24,24,48,72],
                 delta_hours=3, saveto=None, avg_th=3, origin_start="2000-01-01 00:00:00+00:00",debug=False,
                 keep_na=False, verbose=1): 
        '''
            timezones - the time zone of the taken measurements, to be translated to UTC 
            (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
            num_hours_to_avg - the hourly averages will be averaged in these steps
            lags - 
                must include 0 at first. To keep information about future or past.
                The resulting data will be, for each lag: [dust_lag, delta_lag] ([all lags...,all delta...])
            delta_hours - 
                is used for delta calculation. delta_lag = dust_lag - dust_(lag-delta_hours). 
                Defaults to 3.
            saveto - 
                If a string - will save to the resulting dataframe that string. Has to end with ".pkl".
                If None (default) - will not save automatically
            origin_start - the time from which averages of num_hours_to_avg will be calculated
            avg_th - averages of less than this threshold values will be dropped. Defaults to 50% 
            of 6 measurements (for half hourly data, 3 hourly averages)
            debug - for debugging
            keep_na - will not drop na values
        '''     
        self.timezone = timezone
        self.num_hours_to_avg = num_hours_to_avg
        self.lags = lags
        if lags[0] != 0:
            print("Problematic lags list, 0 must be in lags[0] position. Aborting...")
            return
        self.delta_hours = delta_hours
        self.origin_start = origin_start
        self.avg_th = avg_th
        self.debug = debug
        self.keep_na = keep_na
        self.verbose = verbose
        self.saveto = saveto

    def print_dataframe(self, df):
        print(f"{df[:5]}\n...\n{df[-5:]}\nLength: {len(df)}, number of non-NaN values: {df.count()[0]}")
    
    def save_dataframe(self,df,filename=None):
        filename = filename or self.saveto
        if filename is None or filename[-4:]!=".pkl":
            print("Could not save - bad file name. Has to end with .pkl.")
            return
        torch.save(df,filename)
        print(f"Saved dataframe to file {filename}")

    def load_csv_file(self,filename):
        file = open(filename, 'r') 
        return list(csv.reader(file))
    
    def get_data(self):
        # To be implemented in child classes
        raise NotImplementedError
        
    def print_yearly_events_count_per_col(self,df,events_th,col):
        print(f"Yearly events count, threshold={events_th}:")
        first_year,last_year = df.index[0].year,df.index[-1].year
        for y in range(first_year,last_year+1):
            events_count = sum(df[df.index.year==y][col]>=events_th)
            all_count = sum(df[df.index.year==y][col]>=0)
            print(f"{y}:{events_count}, out of {all_count} valid (positive) values")
    
    def check_result(self):
        raise NotImplementedError



    
