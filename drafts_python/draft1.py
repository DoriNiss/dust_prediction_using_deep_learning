import pandas as pd
import numpy as np
import torch

# the times of PMs will be given in Israel time, will have to be converted
times_meteo = [# will be built from filenames / from columns of pms
    "2000-01-01 00:00",
    "2000-01-01 03:00",
    "2000-01-01 06:00",
    "2000-01-01 09:00",
    "2000-01-01 12:00",
    "2000-01-01 15:00",
    "2000-01-01 18:00",
    "2000-01-01 21:00",
    "2000-01-02 00:00",
    "2000-01-02 03:00",
    "2000-01-02 06:00",
    "2000-01-02 09:00",
    "2000-01-02 12:00",
    "2000-01-02 15:00",
    "2000-01-02 18:00",
    "2000-01-02 21:00",
    "2000-07-01 00:00",
    "2000-07-01 03:00",
    "2000-07-01 06:00",
    "2000-07-01 09:00",
    "2000-07-01 12:00",
    "2000-07-01 15:00",
    "2000-07-01 18:00",
    "2000-07-01 21:00",
    "2000-07-02 00:00",
    "2000-07-02 03:00",
    "2000-07-02 06:00",
    "2000-07-02 09:00",
    "2000-07-02 12:00",
    "2000-07-02 15:00",
    "2000-07-02 18:00",
    "2000-07-02 21:00",
]
N = len(times_meteo)
data_meteo = torch.rand([N,3,8,8]) # N 3-hourly measurements of dummy shape 3x8x8
meteo_tensors = data_meteo.split(1,0)
# print(len(meteo_tensors),meteo_tensors[0].shape)

times_meteo_pd_tdi = pd.to_datetime(times_meteo).tz_localize('UTC')

# print("meteo times:",times_meteo_pd_tdi)

pd_meteo_data = pd.DataFrame({
    "date": times_meteo_pd_tdi,
    "meteo tensor": meteo_tensors,
})
print("\nmeteorology")
print(pd_meteo_data.describe())
print(pd_meteo_data[0:1])
print(pd_meteo_data["date"][11:15])
print(pd_meteo_data["meteo tensor"][11].shape)

pd_meteo_data_dateindex = pd.DataFrame(
    # {"meteorology": [t.numpy() for t in meteo_tensors]},
    {"meteorology": meteo_tensors},
    index = pd_meteo_data["date"]
)
print(pd_meteo_data_dateindex["meteorology"][0].shape)
list_of_tensors = [t.squeeze(0) for t in pd_meteo_data_dateindex["meteorology"]]
print(torch.stack(list_of_tensors).shape)

times_dust = [
    "2000-01-01 00:30",
    "2000-01-01 01:00",
    "2000-01-01 01:30",
    "2000-01-01 02:00",
    "2000-01-01 02:30",
    "2000-01-01 03:00",
    "2000-01-01 03:30",
    "2000-01-01 04:00",
    "2000-01-01 04:30",
    "2000-01-01 05:00",
    "2000-01-01 05:30",
    "2000-01-01 06:00",
    "2000-01-01 06:30",
    "2000-01-01 07:00",
    "2000-01-01 07:30",
    "2000-01-01 08:00",
    "2000-01-01 08:30",
    "2000-01-01 09:00",
    "2000-01-01 09:30",
    "2000-01-01 10:00",
    "2000-01-01 10:30",
    "2000-01-01 11:00",
    "2000-01-01 11:30",
    "2000-01-01 12:00",
    "2000-01-01 12:30",
    "2000-01-01 13:00",
    "2000-01-01 13:30",
    "2000-01-01 14:00",
    "2000-01-01 14:30",
    "2000-01-02 10:00",
    "2000-01-02 10:30",
    "2000-01-02 11:00",
    "2000-01-02 11:30",
    "2000-01-02 12:00",
    "2000-01-02 12:30",
    "2000-01-02 13:00",
    "2000-01-02 13:30",
    "2000-01-02 14:00",
    "2000-01-02 14:30",
    "2000-01-02 15:00",
    "2000-07-01 00:30",
    "2000-07-01 01:00",
    "2000-07-01 01:30",
    "2000-07-01 02:00",
    "2000-07-01 02:30",
    "2000-07-01 03:30",
    "2000-07-01 04:00",
    "2000-07-01 04:30",
    "2000-07-01 05:00",
    "2000-07-01 05:30",
    "2000-07-01 06:00",
    "2000-07-01 06:30",
    "2000-07-01 07:00",
    "2000-07-01 07:30",
    "2000-07-01 08:00",
    "2000-07-01 08:30",
    "2000-07-01 09:00",
    "2000-07-01 09:30",
    "2000-07-01 10:00",
    "2000-07-01 10:30",
    "2000-07-01 11:00",
    "2000-07-01 11:30",
    "2000-07-01 12:00",
    "2000-07-01 12:30",
    "2000-07-01 13:00",
    "2000-07-01 13:30",
    "2000-07-01 14:00",
    "2000-07-01 14:30",
    "2000-07-02 10:00",
    "2000-07-02 10:30",
    "2000-07-02 11:00",
    "2000-07-02 11:30",
    "2000-07-02 12:00",
    "2000-07-02 12:30",
    "2000-07-02 13:00",
    "2000-07-02 13:30",
    "2000-07-02 14:00",
    "2000-07-02 14:30",
    "2000-07-02 15:00",
]

times_dust_pd_tdi = (pd.to_datetime(times_dust)
                .tz_localize('Israel')
                .tz_convert('UTC'))

# print("dust times:",times_dust_pd_tdi)

dust_data = torch.rand(len(times_dust))
# print(dust_data.shape)


# pd_meteo_data = pd.DataFrame(meteo_tensors,
#                              index=times_meteo_pd_tdi,
#                              columns=["meteo tensor"])

pd_dust_data = pd.DataFrame({
    "date": times_dust_pd_tdi,
    "dust tensor": dust_data,
})

pd_dust_data.loc[[0,1,2,3,4,5,
                 12,13,14,15,
                 20,21,22,23,
                 40,41,42,43,
                 50,51,52,53,54,55,
                 70,71,72,73],
                 "dust tensor"] = np.nan

print("\ndust")
print(pd_dust_data.describe())
print(pd_dust_data[:10])
print(pd_dust_data[10:20])
print(pd_dust_data[20:30])
print(pd_dust_data[30:40])
print(pd_dust_data[40:50])
print(pd_dust_data[50:60])
print(pd_dust_data[60:70])
print(pd_dust_data[70:])
print(pd_dust_data["date"][11:15])
print(pd_dust_data["dust tensor"][11].shape)

print("\ndust - cleaned")
pd_dust_clean = pd_dust_data.dropna(how="any")
print(pd_dust_data.describe())

print("\nindexing and lags tests")
sample_idx = 10
sample_time = times_dust_pd_tdi[sample_idx]
mask = pd_dust_clean["date"]==sample_time
print(pd_dust_clean[mask])
sample_time_na = pd.to_datetime("2010-01-01 00:01").tz_localize('Israel').tz_convert('UTC')
print(sample_time_na)
mask_na = pd_dust_clean["date"]==sample_time_na
print(pd_dust_clean[mask_na], pd_dust_clean[mask_na].empty)
lags = [pd.Timedelta(lag,unit="h") for lag in [-1,0,1,2]]
sample_time_lags = [sample_time+lag for lag in lags]
print("time:",sample_time,"lags:",sample_time_lags)
print("dust lags tests")

pd_dust_data_dateindex = pd.DataFrame(
    {"dust tensor": dust_data},
    index = times_dust_pd_tdi
)
print(pd_dust_data_dateindex)
pd_dust_data_dateindex["shifted_m1"] = pd_dust_data_dateindex["dust tensor"].shift(periods=1,freq="h")
pd_dust_data_dateindex["lag_m1"] = pd_dust_data_dateindex["dust tensor"]-pd_dust_data_dateindex["shifted_m1"]
print(pd_dust_data_dateindex)



print("\ndust - avgs")
# pd_dust_3h_avgs = pd_dust_data.dropna(how="any")
print(pd_dust_data.describe())
start_dust_date = times_meteo[0]
dust_3h_grouped = pd_dust_clean.groupby(pd.Grouper( # remove the origin argument for pandas older than 1.1, this is only to make sure the times are synced. Default should work as well
    key="date",freq="3h", origin=times_meteo_pd_tdi[0], 
    label="left"))
print("grouped:",dust_3h_grouped.describe())
print("sample rows:\n",dust_3h_grouped.count()[3:7],"\n",dust_3h_grouped.mean()[3:7])
count_th = 3*2/2
dust_3h_grouped_clean = dust_3h_grouped.mean()[dust_3h_grouped.count()>=count_th].dropna(how="any")
print("grouped, cleaned:",dust_3h_grouped_clean.describe())
print(dust_3h_grouped_clean[:])

#                            dust tensor
# date                                  
# 2000-01-01 00:00:00+00:00     0.525851
# 2000-01-01 03:00:00+00:00     0.830765
# 2000-01-01 06:00:00+00:00     0.431951
# 2000-01-01 09:00:00+00:00     0.520570
# 2000-01-02 09:00:00+00:00     0.724012
# 2000-01-02 12:00:00+00:00     0.761043
# 2000-07-01 00:00:00+00:00     0.433519
# 2000-07-01 06:00:00+00:00     0.737689
# 2000-07-01 09:00:00+00:00     0.628034
# 2000-07-02 09:00:00+00:00     0.243801



# make avgs, lags, combine

# lags: -3,0,3 h
print("\nlags")
lags = [0,-3,3] # hours
lag_distance = 3
dust_avgs_lags = pd.DataFrame(
    {"dust_0": dust_3h_grouped_clean["dust tensor"]},
    index=dust_3h_grouped_clean.index
)
print(dust_avgs_lags)
for lag in lags:
    shift_name = f"dust_{lag}" if lag>=0 else f"dust_m{-lag}"
    delta_name = f"delta_{lag}" if lag>=0 else f"delta_m{-lag}"
    dusts_lag = dust_avgs_lags["dust_0"].shift(periods=-lag,freq="h")
    dusts_just_before_lag = dusts_lag.shift(periods=lag_distance,freq="h")
    dust_avgs_lags[shift_name] = dusts_lag
    dust_avgs_lags[delta_name] = dusts_lag-dusts_just_before_lag

dust_avgs_lags = dust_avgs_lags.dropna(how="any")
print(dust_avgs_lags)

# test_tdi = pd.date_range("2011-05-06 00:00", periods=10, freq="3h")
# print("test:",test_tdi)

# dust_tensor = torch.tensor(dust_avgs_lags.to_numpy())
# print(dust_tensor.shape,dust_tensor)

print(pd_meteo_data_dateindex.index, dust_avgs_lags.index)
print("check:",pd_meteo_data_dateindex.index[2],dust_avgs_lags.index[0],pd_meteo_data_dateindex.index[2]==dust_avgs_lags.index[0])
combined_data = pd_meteo_data_dateindex.join(dust_avgs_lags, how="inner")
print(combined_data)
result_dataset_tensor_meteorology = torch.stack([t for t in combined_data["meteorology"].squeeze(0)]) # torch.save...
result_dataset_tensor_dust = torch.tensor([combined_data.iloc[:,1:].to_numpy()]).squeeze(1) # torch.save...
print(result_dataset_tensor_meteorology.shape, result_dataset_tensor_dust.shape)
# metadata_dust = ...
# metadata_meteo = ...

print()

