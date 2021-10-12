#!/usr/bin/env python
# coding: utf-8

import torch


dir_path = "../../data/yearly_splits/"

split1_train_path = dir_path+"split1_train_ordered.pkl"
split2_train_path = dir_path+"split2_train_extreme_ratios_in_train_avg_in_valid.pkl"
split3_train_path = dir_path+"split3_train_extreme_ratios_in_valid_avg_in_train.pkl"
split4_train_path = dir_path+"split4_train_max_num_events_in_train_avg_valid.pkl"
split5_train_path = dir_path+"split5_train_train_distant_years_valid_between.pkl"

split1_valid_path = dir_path+"split1_valid_ordered.pkl"
split2_valid_path = dir_path+"split2_valid_extreme_ratios_in_train_avg_in_valid.pkl"
split3_valid_path = dir_path+"split3_valid_extreme_ratios_in_valid_avg_in_train.pkl"
split4_valid_path = dir_path+"split4_valid_max_num_events_in_train_avg_valid.pkl"
split5_valid_path = dir_path+"split5_valid_train_distant_years_valid_between.pkl"


# resulting_statistics = {
#     2000: {"events": 162 , "clear": 1072 , "total": 1234 , "ratio": 15.11 },
#     2001: {"events": 160 , "clear": 2090 , "total": 2250 , "ratio": 7.66  },
#     2002: {"events": 228 , "clear": 1636 , "total": 1864 , "ratio": 13.94 },
#     2003: {"events": 411 , "clear": 1634 , "total": 2045 , "ratio": 25.15 },
#     2004: {"events": 306 , "clear": 1983 , "total": 2289 , "ratio": 15.43 },
#     2005: {"events": 217 , "clear": 1838 , "total": 2055 , "ratio": 11.81 },
#     2006: {"events": 309 , "clear": 2099 , "total": 2408 , "ratio": 14.72 },
#     2007: {"events": 274 , "clear": 2367 , "total": 2641 , "ratio": 11.58 },
#     2008: {"events": 278 , "clear": 2207 , "total": 2485 , "ratio": 12.6  },
#     2009: {"events": 251 , "clear": 2044 , "total": 2295 , "ratio": 12.28 },
#     2010: {"events": 423 , "clear": 2129 , "total": 2552 , "ratio": 19.87 },
#     2011: {"events": 282 , "clear": 1349 , "total": 1631 , "ratio": 20.9  },
#     2012: {"events": 151 , "clear": 619  , "total": 770  , "ratio": 24.39 },
#     2013: {"events": 292 , "clear": 1881 , "total": 2173 , "ratio": 15.52 },
#     2014: {"events": 176 , "clear": 1722 , "total": 1898 , "ratio": 10.22 },
#     2015: {"events": 198 , "clear": 1707 , "total": 1905 , "ratio": 11.6  },
#     2016: {"events": 246 , "clear": 2055 , "total": 2301 , "ratio": 11.97 },
#     2017: {"events": 273 , "clear": 1910 , "total": 2183 , "ratio": 14.29 },
#     2018: {"events": 278 , "clear": 2074 , "total": 2352 , "ratio": 13.4  },
#     2019: {"events": 334 , "clear": 1618 , "total": 1952 , "ratio": 20.64 },
#     "total":{"events": 5249 ,"clear": 36034, "total": 41283 ,"ratio": 14.57 }
# }

# Split 1: Ordered
# Training:
# events: 3301, clear: 22448, total: 25749, ratio_avg: 15.087499999999999, num years: 12
# Validation:
# events: 1463, clear: 11349, total: 12812, ratio_avg: 12.833333333333334, num years: 6
                    
# Split 2: extreme_ratios_in_train_avg_in_valid
# Training:
# events: 3164, clear: 21009, total: 24173, ratio_avg: 15.9425, num years: 12
# Validation:
# events: 1556, clear: 10774, total: 12330, ratio_avg: 14.481666666666667, num years: 6
                    
# Split 3: extreme_ratios_in_valid_avg_in_train
# Training:
# events: 3038, clear: 22506, total: 25544, ratio_avg: 13.555833333333332, num years: 12
# Validation:
# events: 1761, clear: 9439, total: 11200, ratio_avg: 19.768333333333334, num years: 6
                    
# Split 4: years_train_split4_max_num_events_in_train_avg_valid
# Training:
# events: 3711, clear: 23295, total: 27006, ratio_avg: 16.365, num years: 12
# Validation:
# events: 1227, clear: 10030, total: 11257, ratio_avg: 12.441666666666668, num years: 6
                    
# Split 5: train_distant_years_valid_between
# Training:
# events: 3122, clear: 21716, total: 24838, ratio_avg: 14.64333333333333, num years: 12
# Validation:
# events: 1677, clear: 10229, total: 11906, ratio_avg: 17.593333333333334, num years: 6
                    


# split1 : ordered
years_train_split1 = [y for y in range(2000,2012)]
years_valid_split1 = [y for y in range(2013,2019)]

# split2 : extreme_ratios_in_train_avg_in_valid
years_train_split2 = [
    2001,
    2003,
    2005,
    2007,
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2019,
]
years_valid_split2 = [
    2000,
    2002,   
    2004,
    2006,
    2017,
    2018,
]

# split3 : extreme_ratios_in_valid_avg_in_train
years_train_split3 = [
    2000,
    2002,
    2004,
    2005,
    2006,
    2008,
    2009,
    2013,
    2015,
    2016,
    2017,
    2018,
]
years_valid_split3 = [
    2001,
    2003,
    2010,
    2011,
    2012,
    2019
]

# split4 : max_num_events_in_train_avg_valid
years_train_split4 = [
    2003,
    2004,
    2006,
    2007,
    2008,
    2009,
    2010,
    2011,
    2013,
    2017,
    2018,
    2019
]
years_valid_split4 = [
    2000,
    2002,
    2005,
    2014,
    2015,
    2016,
]

# split5 : train_distant_years_valid_between
years_train_split5 = [
    2000,
    2001,
    2002,
    2003,
    2004,
    2005,
    2006,
    2015,
    2016,
    2017,
    2018,
    2019
]
years_valid_split5 = [
    2008,
    2009,
    2010,
    2011,
    2012,
    2013,
]


torch.save(years_train_split1,split1_train_path)
torch.save(years_valid_split1,split1_valid_path)

torch.save(years_train_split2,split2_train_path)
torch.save(years_valid_split2,split2_valid_path)

torch.save(years_train_split3,split3_train_path)
torch.save(years_valid_split3,split3_valid_path)

torch.save(years_train_split4,split4_train_path)
torch.save(years_valid_split4,split4_valid_path)

torch.save(years_train_split5,split5_train_path)
torch.save(years_valid_split5,split5_valid_path)







