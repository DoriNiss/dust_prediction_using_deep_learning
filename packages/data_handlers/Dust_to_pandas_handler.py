# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import csv
import pytz

class Dust_to_pandas_handler:
    '''
        This class is used for taking a dust file (defaulted to .csv from the Ministry of Env Protection site https://www.svivaaqm.net/)
        and transform it into a pandas dataframe. This df will be used later in combination with
        the ERA5 dataframe of meteorological parameters and from the combined df the dataset 
        will be built
    '''
    def __init__(self, filename, timezone="Asia/Jerusalem", num_hours_to_avg="3h",lags=[0,-24,24,48,72],
                 delta_hours=3, data_type="MEP", saveto=None, avg_th=3, origin_start="2000-01-01 00:00:00+00:00",
                 use_all=True):
        '''
            timezones - the time zone of the taken measurements, to be translated to UTC (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
            num_hours_to_avg - the hours
            lags - must include 0 at first. The lags which the network will use for loss calculations.
            The resulting data will be, for each lag: [dust_lag, delta_lag]
            delta_hours - is used for delta calculation. delta_lag = dust_lag - dust_(lag-delta_hours). 
            Defaults to 3.
            data_type - MEP is a slightly editted .csv file from the MEP site: all the summary
            items are deleted, the dates' column changed to English "date and time", the other
            column is "PM10 µg/m³", and the empty row between the data and title is deleted,
            so the only rows are: first row - titles, the rest - data.
            Note about the MEP format: the MEP date format is "01/01/2000  1:00:00" in Israel time, 
            and every midnight (00:00) the date is for some reason replaced with an int. 
            Values can be numbers or irrelevant strings.
            All of these are to be carefully transformed into TimeIndex and float (or NaN) values.
            saveto - if None, will not save. If a string - will save to that. Has to be a ".pkl"
            origin_start - the time from which averages of num_hours_to_avg will be calculated
            avg_th - averages of less than this threshold values will be dropped. Defaults to 50% 
            of 6 measurements (for half hourly data, 3 hourly averages)
            use_all - for debugging, taking only a few of the csv rows instead of everything
        '''
        self.filename = filename
        self.timezone = timezone
        self.num_hours_to_avg = num_hours_to_avg
        self.lags = lags
        if lags[0] != 0:
            print("Problematic lags list, 0 must be in lags[0] position. Aborting...")
            return
        self.delta_hours = delta_hours
        self.data_type = data_type
        self.origin_start = origin_start
        self.avg_th = avg_th
        self.use_all = use_all
        print("Loading data and creating a pandas DataFrame: ...")
        self.dust_raw = self.get_data()
        print("... Done! Created a pandas DataFrame (times shifted to UTC):",self.dust_raw.describe(),"len:",len(self.dust_raw))
        print("Removing NaN values: ...")
        self.dust_raw = self.dust_raw.dropna(how="any")
        print("... Done! Dropped NaN values:",self.dust_raw.describe(),"len:",len(self.dust_raw))
        print(f"Calculating {self.num_hours_to_avg} hourly averages since {self.origin_start}: ...")
        dust_avgs = self.calculate_averages(self.dust_raw)
        print(f"... Done! Averaged in timebase of {self.num_hours_to_avg}:")
        print(dust_avgs.describe())
        print(f"Calculating lags (dust,delta_dust) at these hours: {lags}")
        self.dust_lags = self.calculate_lags(dust_avgs)
        print(f"... Done! Result is saved in self.dust_lags:")
        print(self.dust_lags.describe())
        print("You can check the result using self.check_result(rows_start, rows_end)")
        if saveto is not None:
            self.saveto(saveto)

    def saveto(self,filename):
        if filename[-4:] == ".pkl":
            self.dust_lags.to_pickle(filename)
            print(f"Saved self.dust_lags to file {filename}")
        else:
            print("Could not save - bad file name. Has to end with .pkl. Use only self.saveto(filename) instead of re-computing everything")

    
    def get_csv_only(self):
        # Used for debugging
        file = open(self.filename, 'r') 
        return list(csv.reader(file))
    
    def get_data(self):
        # can't just pd.read_csv becuase of bad format. Assuming the same format as noted above
        file = open(self.filename, 'r') 
        dust_csv = list(csv.reader(file))
        if self.data_type == "MEP":
            dates, values = self.get_formatted_dates_and_values_from_csv_reader_list(dust_csv[1:])
        return pd.DataFrame({"dust_0":values},index=dates)
        # return pd.read_csv(self.filename)
        
    def get_formatted_dates_and_values_from_csv_reader_list(self, reader_list):
        dates,values = [],[]
        rows_to_use = reader_list if self.use_all else reader_list[:30000]
        for i,row in enumerate(rows_to_use):
            if row[0] == "":
                continue
            date = self.to_foramtted_date(i,reader_list)
            if date is None:
                continue
            row_value = row[1]
            value = self.to_foramtted_value(row_value)
            dates.append(date)
            values.append(value)
        return dates,values

    def to_foramtted_date(self, row_idx, all_rows):
        # returns a TimeIndex date, assuming the bad integers happens every 00:00 only
        date_in_csv_format = all_rows[row_idx][0]
        formatted_date = None
        format_csv = "%d/%m/%Y %H:%M"
        def is_valid_date(str_date):
            try:
                pd.to_datetime(str_date)
                return True
            except ValueError:
                return False
        if is_valid_date(date_in_csv_format):
            formatted_date = pd.to_datetime(date_in_csv_format,format=format_csv)
        else:
            next_date_in_csv = all_rows[row_idx+1][0]
            last_date_in_csv = all_rows[row_idx-1][0]
            date_for_check_in_csv = all_rows[row_idx+2][0]
            if is_valid_date(next_date_in_csv):
                formatted_date = pd.to_datetime(next_date_in_csv,format=format_csv)+pd.Timedelta("-30m")
            elif is_valid_date(last_date_in_csv):
                formatted_date = pd.to_datetime(last_date_in_csv,format=format_csv)+pd.Timedelta("30m")
            # only in this case it will work - otherwise ignore the row:
            if is_valid_date(date_for_check_in_csv):
                if (formatted_date.hour != 0 or formatted_date.day != pd.to_datetime(
                    date_for_check_in_csv,format=format_csv).day):
                    formatted_date = None
            else: formatted_date = None
        try: 
            shifted_utc_time = formatted_date.tz_localize(self.timezone).tz_convert('UTC')
            return shifted_utc_time
        except pytz.NonExistentTimeError:
            # print(formatted_date,formatted_date.day)
            return None          
        except pytz.AmbiguousTimeError:
            # print(formatted_date,formatted_date.day)
            return None          


    def to_foramtted_value(self, value_in_csv):
        def is_valid_value(str_value):
            try:
                float(str_value)
                return True
            except ValueError:
                return False
        return np.nan if not is_valid_value(value_in_csv) else float(value_in_csv) 

    def calculate_averages(self, dust_dataframe):
        dust_grouped = dust_dataframe.groupby(pd.Grouper( # remove the origin argument for pandas older than 1.1, this is only to make sure the times are synced. Default should work as well
            freq=self.num_hours_to_avg, origin=self.origin_start,label="left"))
        dust_avgs = dust_grouped.mean()[dust_grouped.count()>=self.avg_th].dropna(how="any")
        return dust_avgs
    
    def calculate_lags(self, dust_avg):
        dust_avgs_lags = pd.DataFrame(
            {"dust_0": dust_avg["dust_0"]},
            index=dust_avg.index
        )
        for lag in self.lags:
            shift_name = f"dust_{lag}" if lag>=0 else f"dust_m{-lag}"
            delta_name = f"delta_{lag}" if lag>=0 else f"delta_m{-lag}"
            dusts_lag = dust_avgs_lags["dust_0"].shift(periods=-lag,freq="h")
            dusts_just_before_lag = dusts_lag.shift(periods=self.delta_hours,freq="h")
            dust_avgs_lags[shift_name] = dusts_lag
            dust_avgs_lags[delta_name] = dusts_lag-dusts_just_before_lag
        dust_avgs_lags = dust_avgs_lags.dropna(how="any")
        return dust_avgs_lags

    def check_result(self, rows_start, rows_end):
        print("Not implemented yet. You actually can't check the result like that")

    # uplaod csv
    # translate dates to timeindexes
    # create a dataframe with these as index - values that are not numbers -> NaN (numpy)
    # clean_invalid_data -> dropna
    # avgs
    # lags
    # print statistcs



    
