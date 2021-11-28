#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from DustToPandasHandler import *
import numpy as np


dust_csv_filename = "../../data/data_pm10_BeerSheva_20000101_20210630.csv"
result_filename = "../../data/dust_20000101to20213006_3h_7days_future_with_history.pkl"
yearly_dir = "../../data/dust_20000101to20213006_3h_7days_future_with_history"
yearly_base_filename = "dust_dataframe"


# lags is in hours!
lags = [i*24 for i in range(8)]+[-96,-72,-48,-36,-24,-18,-12,-9,-6,-3]
print(lags)


dust_handler = DustToPandasHandler(filename=dust_csv_filename, saveto=result_filename, keep_na=False, 
                                   num_hours_to_avg="3h", avg_th=3, delta_hours=3, lags=lags,
                                   use_all=True)

















idxs = np.arange(10020,10030)
lag = 72
idxs_diff = -lag//3
idxs_lag = idxs+idxs_diff
lag_name = "dust_"+str(lag)
print(dust_handler.dust_lags.index[idxs])
print(dust_handler.dust_lags.index[idxs_lag])
print(dust_handler.dust_lags["dust_0"][idxs])
print(dust_handler.dust_lags[lag_name][idxs_lag])




