#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from DustToPandasHandler import *


dust_csv_filename = "../../data/data_pm10_BeerSheva_20000101_20210630.csv"
result_filename = "../../data/dust_20000101to20213006_6h_keep_na.pkl"


dust_handler = DustToPandasHandler(filename=dust_csv_filename, saveto=result_filename, keep_na=True, 
                                   num_hours_to_avg="6h", avg_th=6, delta_hours=6)


import numpy as np


idxs = np.arange(10000,10010)
dust_handler.dust_lags.index[idxs], dust_handler.dust_lags["dust_0"][idxs]


last_idx = -1
dust_handler.dust_lags.index[last_idx], dust_handler.dust_lags["dust_0"][last_idx]







