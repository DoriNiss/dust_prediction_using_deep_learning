#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/data_handlers')
from MeteorologyToPandasHandler_Super import *


meteo_folder = "/work/meteogroup"

params = {
    "PV310": {"title":"","size":[81,189],"hourly_res":3,"folder_path":f"{meteo_folder}/era5","file_prefix":"S",
              "netCDF_name":"PV"},
    "PV340": {"title":"","size":[81,189],"hourly_res":3,"folder_path":f"{meteo_folder}/era5","file_prefix":"S",
              "netCDF_name":"PV"},
#     "T310": {"title":"","size":[81,189],"hourly_res":3,"folder_path":f"{meteo_folder}/era5","file_prefix":"P",
#               "netCDF_name":"T"},
#     "Q310": {"title":"","size":[81,189],"hourly_res":3,"folder_path":f"{meteo_folder}/era5","file_prefix":"P",
#               "netCDF_name":"Q"},
#     "Z500": {"title":"","size":[81,189],"hourly_res":3,"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P",
#               "netCDF_name":"Z"},
#     "Z850": {"title":"","size":[81,189],"hourly_res":3,"folder_path":f"{meteo_folder}/era5/plev","file_prefix":"P",
#               "netCDF_name":"Z"},
    "PM10": {"title":"","size":[41,95],"hourly_res":6,"folder_path":f"{meteo_folder}/cams","file_prefix":"PM",
              "netCDF_name":"pm10"},
}





handler = MeteorologyToPandasHandler_Super(
    params, dates=None, debug=True, keep_na=False, result_size=[81,189],result_hourly_res=3
)


print(handler.params_to_interpolate)
print(handler.paths_and_params)


handler.print_param_info("PV310")


handler.print_param_varaiable_info("PV310","lat") 


idxs_dict = {'time':[0],'lev':[0],'lat':[0,5,10],'lon':[0]}
handler.set_param_idxs("PV310", idxs_dict, avg_over_idxs=None)
handler.set_param_idxs("PV340", idxs_dict, avg_over_idxs=None)


handler.print_param_info("PM10")


handler.print_param_varaiable_info("PM10","latitude") 


idxs_dict = {'time':[0],'latitude':[0,5,10],'longitude':[0]}
handler.set_param_idxs("PM10", idxs_dict, avg_over_idxs=None)





handler.paths_and_params


start,end = pd.to_datetime("2002-12-30 00:00:00",utc=True),pd.to_datetime("2002-12-30 18:00:00",utc=True)
idxs = handler.get_paths_and_params_idxs(start,end)
print(idxs)


[handler.paths_and_params[i]["timestamp"] for i in idxs]


a = np.array([[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],[[10,20,30],[40,50,60],[70,80,90],[100,110,120]]])
a, a.shape


idxs_dict_test = {0:[0],1:[2,3],2:[0,1]}


# idxs = (slice(v for v in values) for values in idxs_dict_test.values())
# list(idxs)
b = a
for dim in idxs_dict_test.keys():
    b = np.take(b,idxs_dict_test[dim],axis=dim)
b,b.shape


params = handler.load_params_from_path(handler.paths_and_params[0])


handler.params_idxs


alist = [1,1,2,3]
list(set(alist))


t1,t2 = pd.to_datetime("2000-01-01"), pd.to_datetime("2000-01-02")
d1 = {"col1":1,"col2":2,"col3":3}
d2 = {"col4":4,"col5":5,"col6":6}
d3 = {"col4":20,"col5":30}
df1 = pd.DataFrame(d1,index=[t1])
df2 = pd.DataFrame(d2,index=[t2])
df3 = pd.DataFrame(d3,index=[t1])


df_dict = {}
df_dict[t1] = {}
for col,val in d1.items():
    df_dict[t1][col]=val
df_dict[t2] = {}
for col,val in d2.items():
    df_dict[t2][col]=val
for col,val in d3.items():
    df_dict[t1][col]=val
t3 = pd.to_datetime("2000-01-03")
df_dict[t3]={}
df = pd.DataFrame(df_dict.values(),index=df_dict.keys())
df


df.index[:5]


handler.dates


df = handler.create_dataframe(handler.dates[-5],handler.dates[-1])


handler.params_to_interpolate


df["PV310"][0].shape


timestep_pd = pd.Timedelta("3h")
time = df.index[1]
time+timestep_pd==df.index[2]

















df["PV310"].values[-1]-df["PV310"].values[0]


df_null_big = pd.DataFrame({},index=pd.to_datetime(["2002-12-31 01:00"],utc=True))
df_test_big = df.join(df_null_big,how="outer")
df_test_big





df_test_big["PV310"].interpolate().values





df_test = pd.DataFrame({"A":[0,10],"B":[10,20],"C":[1,2]},index=pd.to_datetime(["2000-01-01","2000-01-04"],utc=True))
df_null = pd.DataFrame({},index=pd.to_datetime(["2000-01-02","2000-01-03"],utc=True))
df_test=df_test.join(df_null,how="outer")
df_test


df_test[cols].interpolate()#.values#.ravel().tolist()


def interpolate_df_rows_linear(df,cols):
    start,end = df[cols].values[0],df[cols].values[-1]
    diff = (end-start)/len(df)
    for i,col in enumerate(df.cols):
        vals = df[col].values
        
        row[cols] = start+i*diff


cols=["A","C"]
print(df_test[cols].values[-1]-df_test[cols].values[0])
interpolate_df_rows_linear(df_test,cols)
df_test


df_test[cols].values.shape


b = a*10


idxs_dict


np.concatenate(df_test_big["PV310"].values)


df_test_big_values_as_array = np.array([v[0] for v in df_test_big.values])
df_test_big_values_as_array.shape,df_test_big_values_as_array


mi = pd.MultiIndex.from_product(df_test_big.index+list(idxs_dict.values()), names=["timestamp"]+list(idxs_dict.keys()))

df_multi = pd.Series(index=mi, data=np)


# pd.MultiIndex.from_arrays(list(idxs_dict.values()))


































