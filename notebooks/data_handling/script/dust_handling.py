#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from Dust_to_pandas_handler import *


dust_csv_filename = "../../data/data_pm10_BeerSheva_20000101_20210630.csv"
result_filename = "../../data/dust_lags_20000101to20213006_0_m24_24_48_72.pkl"


dust_handler = Dust_to_pandas_handler(filename=dust_csv_filename, saveto=result_filename)




