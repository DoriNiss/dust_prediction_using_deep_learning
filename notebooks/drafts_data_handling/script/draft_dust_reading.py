#!/usr/bin/env python
# coding: utf-8

import sys, os
# print(sys.path)
# print()
packages_dir = os.getcwd()+"/../../packages"
data_dir = packages_dir+"/../data/"
sys.path.insert(0 ,packages_dir)

from data_handlers.Dust_to_pandas_handler import * 
import glob
dust_csv_filename = glob.glob(data_dir+"*.csv")





saveto = data_dir+"dust_lags_20000101to20213006_0_m24_24_48_72.pkl"
dust_handler = Dust_to_pandas_handler(filename=dust_csv_filename[0], saveto=saveto)
# dust_handler = Dust_to_pandas_handler(filename=dust_csv_filename[0], use_all=False)
csv_file = dust_handler.get_csv_only()
# 2000-04-14 02:00:00 14
# 2000-04-14 02:30:00 14
# 2000-10-06 00:00:00 6
# 2000-10-06 00:30:00 6
# 2001-04-09 01:00:00 9
# 2001-04-09 01:30:00 9


# dust_handler.saveto(saveto)


dust_handler.dust_lags[4:10]


sample_dates = dust_handler.dust_lags[10:15].index
empty_df = pd.DataFrame({},index=sample_dates)


empty_df["A"] = [1,2,3,4,5]
empty_df


csv_file[-10][1]


import pandas as pd

date_csv_format = '01/01/2000 0:30'
print(pd.to_datetime(date_csv_format).tz_localize('UTC'))


bad_date = '36528'

def isint(str_value):
    try:
        int(str_value)
        return True
    except ValueError:
        return False

print(isint(date_csv_format))


def is_valid_date(str_date):
    try:
        pd.to_datetime(str_date)
        return True
    except ValueError:
        return False
print(is_valid_date(bad_date),is_valid_date(date_csv_format))


print(pd.to_datetime(date_csv_format).tz_localize('UTC')+pd.Timedelta("-30m"))


test_date1 = pd.to_datetime(csv_file[50015][0])
test_date2 = pd.to_datetime(csv_file[50017][0])
print(test_date1, test_date2)
test_date1 = csv_file[50015][0]
test_date2 = csv_file[50017][0]
print(test_date1, test_date2)


"07/11/2002 23:30"
test_date1 = pd.to_datetime(csv_file[50015][0], format="%d/%m/%Y %H:%M")
test_date2 = pd.to_datetime(csv_file[50017][0], format="%d/%m/%Y %H:%M")
print(test_date1, test_date2)


float(csv_file[50000][1])


dust_handler.get_formatted_dates_and_values_from_csv_reader_list(csv_file[50010:50050])


csv_file[50010:50050]


dust_handler.get_formatted_dates_and_values_from_csv_reader_list(csv_file[20010:20050])


csv_file[20010:20050]


dust_handler.dust_raw[20010:20050]


len(dust_handler.dust_raw)


dust_handler.dust_raw[4990:5040]


dust_handler.dust_raw[13380:13420]


dust_handler.dust_raw[22250:22300]


# data was compared with the original csv and found to be correct




