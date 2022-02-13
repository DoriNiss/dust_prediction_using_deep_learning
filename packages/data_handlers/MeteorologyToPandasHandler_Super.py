import glob
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch
from scipy import interpolate
from joblib import Parallel, delayed #conda install -c anaconda joblib
import scipy.ndimage
from PIL import Image

class MeteorologyToPandasHandler_Super:
    def __init__(self, params, dates=None, debug=False, keep_na=False, result_size=[81,189],
                 result_hourly_res=3, params_are_2d=True, infill_missing_res=-9999, infill_missing_params=-8888,
                 infill_missing_values=-7777):
        '''
            params: a dict of dicts contains the needed information to load paramters, interpolate and upsample 
                if needed. One item per parameter level (e.g. Z500 and Z250 will have a separeted row)
                "param" (short title for dataframe's column, e.g. "Z500"): 
                    {
                        "title": short description for graphs
                        "size": size of image, if != result_size, will be upsampled accordingly
                        "hourly_res": e.g. 3 or 6, if != result_hourly_res, will be interpolated accordingly
                        "folder_path": file will be loaded from the path string:
                            f"{folder_path}/{date string, e.g. 2020/01}/{file_prefix}"
                            if multiple parameters are to be loaded from the same path string, they will be
                            loaded once for effeciency, but each has to have its own parameter dict
                        "file_prefix": same
                        "netCDF_name": name to use for loading file[netCDF_name]
                    }
                    each param's idxs will be set individualy
            params_are_2d: will squeeze loaded parameters to 2d np.array (after averaging if defined)
            infill_missing_res: replaces NaN if failed to interpolate in missing hourly resolution (e.g. end of df)
            infill_missing_values: replaces NaN for missing values (e.g. no param in certain year)
            
        '''
        self.infill_missing_res = infill_missing_res
        self.infill_missing_params = infill_missing_params
        self.infill_missing_values = infill_missing_values
        self.params = params
        self.params_are_2d = params_are_2d
        self.debug = debug
        self.keep_na = keep_na
        self.result_size = result_size
        self.result_hourly_res = result_hourly_res
        self.dates = dates if dates is not None else self.get_default_dates()
        self.params_idxs = {}
        self.params_to_avg_idxs = {} 
        print(f"Initiating paths for \n{self.dates[:5]}\n...\n{self.dates[-5:]}: ...")
        self.init_paths_and_params()
        self.init_interpolation_info()
        print("... Done initializing handler!")
        print("Use self.print_param_info(param) to see available varaiables per parameter")
        print("Use self.print_param_varaiable_info(param,v) to see exact indices per variable")
        print("Use self.set_param_idxs(param, idxs_dict, avg_over_idxs=None) to set the parameter's indices,")
        print("define idxs_dict with the parameter's indices, e.g. {'time':[0],'lev':[0],'lat':[0],'lon':[0]},")
        print("use avg_over_idxs to avg over sepcific idx, e.g. ['lev'].")
        print("Use self.create_yearly_dataframes(base_filename,years=None,add_year_to_name=True)"\
              "to load and save yearly data as dataframes")
        print("Use self.create_yearly_dataframes(base_filename,years=None,add_year_to_name=True,njobs=1)"\
              "to do it paralelly")
        print("Use self.create_and_save_yearly_dataframe_batches(\n"\
              "   base_filename,year,add_year_to_name=True,batch_size=30,batch_start=None,batch_end=None) "\
              "to do it even faster (batch_end is not included, defaults to last batch) ")
        print("Or self.create_yearly_dataframes_parallel_batches(\n"\
              "   base_filename,years=None,add_year_to_name=True,njobs=1,"\
              "batch_size=30,num_batch_start=None,num_batch_end=None)")

    def get_default_dates(self):
        if self.debug: start,end = ("2002-12-30 00:00","2003-01-02 18:00")
        else: start,end = ("2000-01-01 00:00","2020-12-31 23:00") 
        dates = pd.date_range(start=start, end=end, tz="UTC", freq=f"{self.result_hourly_res}h")
        return dates
   
    def init_params_to_take_from_same_paths(self):
        self.params_to_take_from_same_paths = []
        params_done,params_left = [],list(self.params.keys())
        for param in self.params.keys():
            if param in params_done: continue
            params_from_same_path = [param]
            param_dict = self.params[param]
            folder_path,file_prefix = param_dict["folder_path"],param_dict["file_prefix"]
            params_left.remove(param)        
            params_done.append(param)
            params_left_frozen = params_left.copy()
            for other_param in params_left_frozen:
                if other_param not in params_left: continue
                other_param_dict = self.params[other_param]
                if folder_path==other_param_dict["folder_path"] and file_prefix==other_param_dict["file_prefix"]:
                    params_from_same_path.append(other_param)
                    params_left.remove(other_param)        
                    params_done.append(other_param)
            self.params_to_take_from_same_paths.append({
                "folder_path":folder_path,"file_prefix":file_prefix,"params":params_from_same_path})
            
    def init_all_dates_strings(self):
        def get_path_strings_from_datetime(t): # e.g. <folder...>/2003/03/P20031212_12*
            y = str(t.year)
            m = str(t.month) if t.month>=10 else "0"+str(t.month)
            d = str(t.day) if t.day>=10 else "0"+str(t.day)
            h = str(t.hour) if t.hour>=10 else "0"+str(t.hour)
            return y,m,d,h
        self.all_dates_strings = []
        for date in self.dates:
            y,m,d,h = get_path_strings_from_datetime(date)
            self.all_dates_strings.append(
                {"date_path_string":f"{y}/{m}", "date_file_string":f"{y}{m}{d}_{h}",
                 "date_pd_string":f"{y}-{m}-{d} {h}:00"})

    def init_paths_and_params(self):
        self.init_params_to_take_from_same_paths()
        self.init_all_dates_strings()
        self.paths_and_params = []
        for params_from_same_path in self.params_to_take_from_same_paths:
            print(f"Initializing paths for {params_from_same_path['params']}: ...")
            for date_strings in tqdm(self.all_dates_strings):
                p,d = params_from_same_path,date_strings
                path_str = f"{p['folder_path']}/{d['date_path_string']}/{p['file_prefix']}{d['date_file_string']}*"
                path = glob.glob(path_str)
                path = "" if path==[] else path[0]
                self.paths_and_params.append({"path":path,"params":p["params"],
                                              "timestamp":pd.to_datetime(d['date_pd_string'],utc=True)})
        print("... Done initializing paths!")
        
    def print_param_info(self,param):
        file = self.load_sample_param_file(param)
        print(f"Full file variables of {param}:")
        print(file.variables)

    def load_sample_param_file(self,param):
        sample_path = self.get_valid_param_path(param)
        if sample_path is None:
            print(f"Error! No path found for {param}. Please retry setting paths, dates or params. Aborting...")
            return
        print(f"Sample path: {sample_path}")
        return self.load_full_netcdf_file(sample_path)
        
    def get_valid_param_path(self,param):
        for path_and_params in self.paths_and_params:
            if param in path_and_params["params"] and path_and_params["path"]!= "":
                return path_and_params["path"]
        return None
        
    def load_full_netcdf_file(self,path):
        if path=="":
            return None 
        try:
            return Dataset(path)
        except:
            return self.infill_missing_values

    def print_param_varaiable_info(self,param,v):
        file = self.load_sample_param_file(param)
        print(f"Indices values of {param}[{v}] (value  [idx]):")
        values = file[v][:].data
        print([f"{value} [{i}]" for i,value in enumerate(values)])
            
    def set_param_idxs(self, param, idxs_dict, avg_over_idxs=None):
        self.params_idxs[param] = idxs_dict
        if avg_over_idxs is not None: self.params_to_avg_idxs[param]=avg_over_idxs
        file = self.load_sample_param_file(param)
        print(f"Param: {param}. The values of each key are set to:")
        for value_key in idxs_dict.keys():
            print("*** Key:", value_key)
            values = file[value_key][idxs_dict[value_key]]
            print("Indices values (value  [idx]):")
            print([f"{value} [{i}]" for i,value in enumerate(values)])
        shape_dict = idxs_dict.copy()
        if avg_over_idxs is not None: shape_dict[avg_over_idxs]=[1]
        print(f"Result shape: {[len(idxs) for idxs in shape_dict.values()]}")
        
    def init_interpolation_info(self):
        self.params_to_interpolate = {"by_time":[], "by_size":[]}
        for param in self.params.keys():
            if self.params[param]["hourly_res"] != self.result_hourly_res:
                self.params_to_interpolate["by_time"].append(param)
            if self.params[param]["size"] != self.result_size:
                self.params_to_interpolate["by_size"].append(param)
        print(f"Initiated interpolation infomation: {self.params_to_interpolate}")
        
    def create_dataframe(self,timestamps_start,timestamps_end):
        paths_and_params_idxs = self.get_paths_and_params_idxs(timestamps_start,timestamps_end)
        paths_and_params_of_dates = [self.paths_and_params[i] for i in paths_and_params_idxs]
        dataframe_as_dict = {}
        for path_and_params in tqdm(paths_and_params_of_dates):
            path_loaded_params =  self.load_params_from_path(path_and_params)
            t = path_and_params["timestamp"]
            params_titles = path_and_params["params"]
            if t not in dataframe_as_dict.keys():
                dataframe_as_dict[t] = {}
            for param,param_title in zip(path_loaded_params,params_titles):
                dataframe_as_dict[t][param_title]=param
        dataframe = pd.DataFrame(dataframe_as_dict.values(),index=dataframe_as_dict.keys())
        dataframe = self.interpolate_df_to_hourly_res(
            dataframe,self.result_hourly_res,self.params_to_interpolate["by_time"])
        dataframe = dataframe.fillna(self.infill_missing_res)
        dataframe = self.infill_df_with_missing_params(dataframe)
        dataframe = dataframe.fillna(self.infill_missing_values) # just in case
        return dataframe
    
    def infill_df_with_missing_params(self,df):
        infill_list = [self.infill_missing_params]*len(df)
        for param in self.params.keys():
            if param not in df.columns:
                df[param]=infill_list
        return df
            
    def get_paths_and_params_idxs(self,timestamps_start,timestamps_end):
        idxs = []
        for idx,path_and_params in enumerate(self.paths_and_params):
            timestamp = path_and_params["timestamp"]
            if timestamp>=timestamps_start and timestamp<=timestamps_end:
                idxs.append(idx)
        return idxs     
        
    def load_params_from_path(self,path_and_params):
        file = self.load_full_netcdf_file(path_and_params["path"])
        if file is None: return [] 
        if file==self.infill_missing_values: return [self.infill_missing_values]*len(path_and_params["params"])
        loaded_params = []
        for param in path_and_params["params"]:
            idxs_from_file_dict = self.params_idxs[param]
            try:
                param_from_file = file[self.params[param]["netCDF_name"]]
            except:
                print(f"Could not find {self.params[param]['netCDF_name']} ({param}) in {path_and_params['path']}")
                continue
            # Assuming idxs_from_file_dict is ordered
            for dim,dim_name in enumerate(idxs_from_file_dict.keys()):
                param_from_file = np.take(param_from_file,idxs_from_file_dict[dim_name],axis=dim)
            if param in self.params_to_avg_idxs.keys():
                param_from_file = np.mean(param_from_file,
                    axis=list(idxs_from_file_dict.keys()).index(self.params_to_avg_idxs[param]))
            if self.params_are_2d: # Assuming the last 2 dims are lat and lon
                for i in range(len(param_from_file.shape)-2):
                    param_from_file=param_from_file[0]
            if param in self.params_to_interpolate["by_size"]:
                param_from_file = self.upsample_param(param_from_file) 
            loaded_params.append(param_from_file)
        return loaded_params
        
    def upsample_param(self,original): 
        upsampled = scipy.ndimage.zoom(original, 2, order=3)
        upsampled = Image.fromarray(upsampled).resize([self.result_size[-1],self.result_size[-2]])
        upsampled = np.array(upsampled)
        return upsampled
    
    def interpolate_df_to_hourly_res(self,df,timestep,params): # TBD vectorization with pd.interpolate()...
        """
            Currently only linear (mean) interpolation is implemented, assuming only 6 or 3 hourly res available
            Interpolates only if NaN between valid rows
        """
        params_existing = [p for p in params if p in df.columns]
        if params_existing == []:
            return df
        params = params_existing
        timestep_pd = pd.Timedelta(f"{timestep}h")
        for row_idx in range(1,len(df)-1):
            row = df[row_idx:row_idx+1]
            time = df.index[row_idx]
            if self.does_row_have_nan(df,row_idx-1) or self.does_row_have_nan(df,row_idx+1):
                continue
            if df.index[row_idx-1]!=time-timestep_pd or df.index[row_idx+1]!=time+timestep_pd:
                continue
            for p in params:
                df[p][row_idx] = 0.5*(df[p][row_idx-1]+df[p][row_idx+1])
        return df    

    def does_row_have_nan(self,df,row_idx):
        has_nan = df[row_idx:row_idx+1].isnull().any(axis=1)[0]
        return has_nan

    def create_and_save_yearly_dataframe(self,base_filename,year,add_year_to_name=True):
        start = pd.to_datetime(f"{year}-01-01 00:00",utc=True)
        end = pd.to_datetime(f"{year}-12-31 23:00",utc=True)
        yearly_df = self.create_dataframe(start,end)
        filename = f"{base_filename}_{year}.pkl" if add_year_to_name else f"{base_filename}.pkl" 
        torch.save(yearly_df,filename)
        print(f"Saved dataframe of year {year}, length {len(yearly_df)} to {filename}")
    
    def create_and_save_yearly_dataframe_monthly(self,base_filename,year,add_year_to_name=True):
        start = pd.to_datetime(f"{year}-01-01 00:00",utc=True)
        end = pd.to_datetime(f"{year}-12-31 23:00",utc=True)
        all_year_timestamps = pd.date_range(start=start,end=end,freq=f'{self.result_hourly_res}h',tz="UTC")
        for m in range(1,12):
            timestamps = all_year_timestamps[all_year_timestamps.month==m]
            if timestamps.empty:
                print(f"Couldn't create data for {year}-{m}")
                continue
            monthly_df = self.create_dataframe(timestamps[0],timestamps[-1])
            filename = f"{base_filename}_{year}_{m}.pkl" if add_year_to_name else f"{base_filename}_{m}.pkl" 
            torch.save(monthly_df,filename)
            print(f"Saved dataframe for {m}-{year}, length {len(monthly_df)}")

    def create_and_save_yearly_dataframe_batches(self,base_filename,year,add_year_to_name=True,batch_size=30,
                                                 num_batch_start=None,num_batch_end=None):
        start = pd.to_datetime(f"{year}-01-01 00:00",utc=True)
        end = pd.to_datetime(f"{year}-12-31 23:00",utc=True)
        all_year_timestamps = pd.date_range(start=start,end=end,freq=f'{self.result_hourly_res}h',tz="UTC")
        num_batches=len(all_year_timestamps)//batch_size
        b_start = num_batch_start or 0
        b_end = num_batch_end or num_batches+1
        for b in range(b_start,b_end):
            if (b+1)*batch_size>=len(all_year_timestamps):
                timestamps = all_year_timestamps[b*batch_size:]
            else:
                timestamps = all_year_timestamps[b*batch_size:(b+1)*batch_size]
            if timestamps.empty:
                print(f"Couldn't create data for {year}-{b}")
                continue
            batch_df = self.create_dataframe(timestamps[0],timestamps[-1])
            filename = f"{base_filename}_{year}_b{b}.pkl" if add_year_to_name else f"{base_filename}_b{b}.pkl" 
            torch.save(batch_df,filename)
            print(f"Saved dataframe for {year}, batch #{b} ({b+1}/{b_end}), length {len(batch_df)}")
            
    def save_metadata(self,base_filename):
        metadata = {"params":self.params,"params_idxs":self.params_idxs,
                    "infill_missing_res":self.infill_missing_res,
                    "infill_missing_params":self.infill_missing_params,
                    "infill_missing_values":self.infill_missing_values,
        }
        torch.save(metadata,f"{base_filename}_metadata.pkl")
    
    def create_yearly_dataframes(self,base_filename,years=None,add_year_to_name=True):
        years_to_calculate = years or list(set([t.year for t in self.dates]))
        if years_to_calculate==[]: return
        self.save_metadata(base_filename)
        for year in years_to_calculate:
            self.create_and_save_yearly_dataframe(base_filename,year,add_year_to_name=add_year_to_name)
    
    def create_yearly_dataframes_parallel(self,base_filename,years=None,add_year_to_name=True,njobs=1):
        years_to_calculate = years or list(set([t.year for t in self.dates]))
        if years_to_calculate==[]: return
        self.save_metadata(base_filename)
        Parallel(n_jobs=njobs,verbose=100)(delayed(self.create_and_save_yearly_dataframe)(
            base_filename,year,add_year_to_name=add_year_to_name) 
            for year in years_to_calculate
        )
            
    def create_yearly_dataframes_parallel_monthly(self,base_filename,years=None,add_year_to_name=True,njobs=1):
        years_to_calculate = years or list(set([t.year for t in self.dates]))
        if years_to_calculate==[]: return
        self.save_metadata(base_filename)
        Parallel(n_jobs=njobs,verbose=100)(delayed(self.create_and_save_yearly_dataframe_monthly)(
            base_filename,year,add_year_to_name=add_year_to_name) 
            for year in years_to_calculate
        )

    def create_yearly_dataframes_parallel_batches(self,base_filename,years=None,add_year_to_name=True,njobs=1,
                                                  batch_size=30,num_batch_start=None,num_batch_end=None):
        years_to_calculate = years or list(set([t.year for t in self.dates]))
        if years_to_calculate==[]: return
        self.save_metadata(base_filename)
        Parallel(n_jobs=njobs,verbose=100)(delayed(self.create_and_save_yearly_dataframe_batches)(
            base_filename,year,add_year_to_name=add_year_to_name,batch_size=batch_size,
            num_batch_start=num_batch_start,num_batch_end=num_batch_end)
            for year in years_to_calculate
        )

    @staticmethod
    def print_varaiables(path):
        sample_file = Dataset(path)
        print(sample_file.variables)


