#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from DustToPandasHandler import *


dust_csv_filename = "../../data/data_pm10_BeerSheva_20000101_20210630.csv"
result_filename = "../../data/dust_20000101to20213006_6h_7days_before_and_after.pkl"
yearly_dir = "../../data/dust_20000101to20213006_6h_7days_before_and_after"
yearly_base_filename = "dust_dataframe"


lags = [i for i in range(0,169,6)] + [-i for i in range(168,0,-6)]
print(lags, lags[:29])
#             split_dust_from_delta - True will set the first cols to be dust and only then deltas, e.g.
#             [dust_0, dust_6, dust_12, delta_0, delta_6, delta_12]


dust_handler = DustToPandasHandler(filename=dust_csv_filename, saveto=result_filename, keep_na=True, 
                                   num_hours_to_avg="6h", avg_th=6, delta_hours=6, lags=lags)


import numpy as np


idxs = np.arange(10000,10010)
lag = 168
idxs_diff = -lag//6
idxs_lag = idxs+idxs_diff
lag_name = "dust_"+str(lag)
print(dust_handler.dust_lags.index[idxs])
print(dust_handler.dust_lags["dust_0"][idxs])
print(dust_handler.dust_lags[lag_name][idxs_lag])


last_idx = -1
dust_handler.dust_lags.index[last_idx], dust_handler.dust_lags["dust_0"][last_idx], dust_handler.dust_lags["dust_168"][last_idx]





for y in range(2000,2022):
    dust_df = dust_handler.dust_lags
    dust_df = dust_df[dust_df.index.year==y]
    torch.save(dust_df,yearly_dir+"/"+yearly_base_filename+"_"+str(y)+".pkl")


# print(dust_handler.dust_lags[dust_handler.dust_lags.index.year==2003])




