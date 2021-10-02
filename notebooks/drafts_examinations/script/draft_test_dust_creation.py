#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from Dust_to_pandas_handler import *

dust_csv_filename = "../../data/dust_from_MEP/data_pm10_BeerSheva_20000101_20210630.csv"


dust_handler = Dust_to_pandas_handler(filename=dust_csv_filename, saveto=None)


print(dust_handler.filename)
print(dust_handler.timezone)
print(dust_handler.num_hours_to_avg)
print(dust_handler.lags)
print(dust_handler.delta_hours)
print(dust_handler.data_type)
print(dust_handler.origin_start)
print(dust_handler.avg_th)
print(dust_handler.use_all)


dust_handler.dust_raw.shape


dust_handler.dust_raw.dropna(how="any").shape


# checking self.calculate_averages(self.dust_raw)


raw = dust_handler.dust_raw


len(raw), len(raw)/6


start = dust_handler.origin_start
hours_to_avg = dust_handler.num_hours_to_avg
start, hours_to_avg


dust_grouped = raw.groupby(pd.Grouper( 
    freq=hours_to_avg, origin=start,label="left"))


dust_grouped.describe()


raw_chunk = raw[11110:11120]
print(len(raw_chunk), len(raw_chunk)/6)
raw_chunk


dust_grouped_chunk = raw_chunk.groupby(pd.Grouper( 
    freq=hours_to_avg, origin=start,label="left"))
dust_grouped_chunk.describe()


dust_grouped_chunk.mean()[dust_grouped_chunk.count()>=3].dropna(how="any")


raw_with_na = dust_handler.get_data()
len(raw_with_na)


376761/6


dust_grouped_with_na = raw_with_na.groupby(pd.Grouper( 
    freq=hours_to_avg, origin=start,label="left"))


len(dust_grouped_with_na)


dust_grouped_with_na.mean()[dust_grouped_with_na.count()>=3].dropna(how="any")
len(dust_grouped_with_na)


# vars(dust_grouped_with_na)
# dust_grouped_with_na.obj[111110:111120]


sample_idx = 311167
# sample_idx = 111147
raw_with_na[sample_idx:sample_idx+48]


avgs_with_na = dust_grouped_with_na.mean()[dust_grouped_with_na.count()>=3].dropna(how="any")


sample_idx = 43272
avgs_with_na[sample_idx:sample_idx+10]


len(raw_with_na), len(raw_with_na)/6


len(avgs_with_na)




