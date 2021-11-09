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

# +
class MeteorologyToPandasHandler:
    def __init__(self, params=None, folders=None, prefixes=None, netcdf_keys=None,
        dates=None, debug=False, keep_na=False, add_cams=True, upsample_to=[81,169]):
        '''
            Used for loading meteorology params from a server (defaults values regards to the Chemfarm of WIS)
            and transform them into a pandas DataFrame
            dates - a DateTime list of wanted dates (ERA5 has data every 3 hours - assuming all
            hours of each day are wanted)
            Usage:
            1. Init with Meteorology_to_pandas_handler() with the wanted arguments
            2. .print_param_info(param) - this will help choosing the wanted indices
            3. .set_idxs(param) - usually times will just be np.array([0]) - for each param!
            4. your_dataframe = self.load_data() - returns dataframe of {param:list[numpy arrays]}
            5. save your dataframe with self.save_dataframe(dataframe, filename)
            CAREFULL - going to be heavy and will take a while...

        '''
        self.keep_na = keep_na
        self.add_cams = add_cams
        self.debug = debug
        self.paths = {}
        self.params_to_take_from_paths = {}
        self.param_idxs = {}
        self.param_shapes = {}
        self.upsample_to = upsample_to
        self.params = params or self.get_default_params()
        self.folders = folders or self.get_default_folders()
        self.prefixes = prefixes or self.get_default_prefixes()
        self.netcdf_keys = netcdf_keys or self.get_default_netcdf_keys()
        self.dates = dates or self.get_default_dates()
        self.init_paths()
        self.init_params_to_take_from_paths()
        self.init_param_shapes()

    def get_default_params(self):
        default_list = ["SLP", "Z", "U", "V", "PV"]
        if self.add_cams: 
            default_list+=["aod550","duaod550","aermssdul","aermssdum","u10","v10"]
        return default_list

    def get_default_folders(self):
        default_dict = {
            "SLP":"/work/meteogroup/era5/",
            "Z":"/work/meteogroup/era5/plev/",
            "U":"/work/meteogroup/era5/plev/",
            "V":"/work/meteogroup/era5/plev/",
            "PV":"/work/meteogroup/era5/isn/",
        }
        if self.add_cams: 
            default_dict["aod550"] = "/work/meteogroup/cams/"
            default_dict["duaod550"] = "/work/meteogroup/cams/"
            default_dict["aermssdul"] = "/work/meteogroup/cams/"
            default_dict["aermssdum"] = "/work/meteogroup/cams/"
            default_dict["u10"] = "/work/meteogroup/cams/"
            default_dict["v10"] = "/work/meteogroup/cams/"
        return default_dict

    def get_default_prefixes(self):
        default_dict = {"PV":"P","Z":"P","U":"P","V":"P","SLP":"P"}
        if self.add_cams: 
            default_dict["aod550"] = "D"
            default_dict["duaod550"] = "D"
            default_dict["aermssdul"] = "D"
            default_dict["aermssdum"] = "D"
            default_dict["u10"] = "D"
            default_dict["v10"] = "D"
        return default_dict

    def get_default_netcdf_keys(self):
        default_dict = {
            "SLP":["time", "lat", "lon"],
            "Z":["time", "lev", "lat", "lon"],
            "U":["time", "lev", "lat", "lon"],
            "V":["time", "lev", "lat", "lon"],
            "PV":["time", "lev", "lat", "lon"],
        }
        if self.add_cams: 
            default_dict["aod550"] = ["time", "latitude", "longitude"]
            default_dict["duaod550"] = ["time", "latitude", "longitude"]
            default_dict["aermssdul"] = ["time", "latitude", "longitude"]
            default_dict["aermssdum"] = ["time", "latitude", "longitude"]
            default_dict["u10"] = ["time", "latitude", "longitude"]
            default_dict["v10"] = ["time", "latitude", "longitude"]
        return default_dict

    def get_default_dates(self):
        if self.debug:
            start,end = "2002-12-25 00:00","2003-01-05 18:00"
        else:
            start,end = "2000-01-01 00:00","2021-12-31 18:00"
        freq = "6h" if self.add_cams else "3h"
        dates = pd.date_range(start=start, end=end, tz="UTC", freq=freq)
        return dates
   
    def init_paths(self):
        def get_path_strings_from_datetime(t): # e.g. <folder...>/2003/03/P20031212_12*
            y = str(t.year)
            m = str(t.month) if t.month>=10 else "0"+str(t.month)
            d = str(t.day) if t.day>=10 else "0"+str(t.day)
            h = str(t.hour) if t.hour>=10 else "0"+str(t.hour)
            return y,m,d,h
        for param in self.params:
            print(f"Initializing paths of {param}")
            self.paths[param] = []
            path = ""
            for date in tqdm(self.dates):
                y,m,d,h = get_path_strings_from_datetime(date)
                path_str = self.folders[param]+y+"/"+m+"/"+self.prefixes[param]+y+m+d+"_"+h+"*"
                path = glob.glob(path_str)
                if path == []:
                    print(f"No file for {date}, will add NaN")
                    path.append("")
                self.paths[param].append(path[0])
        # Paths' lengths sanity check:
        paths_len = len(self.dates)
        for p in self.params:
            if paths_len != len(self.paths[p]):
                print("Error! Failed paths' lengths sanity check")
                return
        print("Passed paths' lengths sanity check succefully!")

    def does_date_exist_in_all_params(self,date,date_idx):
        for param in self.params:
            if self.paths[param][date_idx]== "": # meaning no date was found when init_paths()
                return False
        return True
    
    def init_params_to_take_from_paths(self):
        """
            Run through all dates until a valid date is found (is_date_exist_in_all_params==True),
            set self.params_to_take_from_paths to {date:{path:[params_to_be_taken]}}. 
            Increases effeciency of load_data()
        """
        print("Initiating parameters to take from each path...")
        for i,date in enumerate(tqdm(self.dates)):
            dict_of_date = {}
            self.params_to_take_from_paths[date] = dict_of_date    
            if not self.does_date_exist_in_all_params(date,i):
                continue
            # Init dict of paths:
            for p in self.params:
                p_path = self.paths[p][i]
                dict_of_date[p_path]=[]
            # Populate with list of parameters in their corresponding path
            for p in self.params:
                p_path = self.paths[p][i]
                dict_of_date[p_path]+=[p]
            self.params_to_take_from_paths[date] = dict_of_date    
        print("...Done initiating parameters to take from each path")
            
    def init_param_shapes(self):
        good_idx = self.get_first_valid_date_idx()
        if good_idx is None:
            print("Error! No date that exists in all parameters was found. Please retry initializing")
            return
        for p in self.params:
            sample_file = self.load_full_netcdf_file(self.paths[p][good_idx])
            self.param_shapes[p] = sample_file[p].shape
        print(f"Initialized params' shapes: {self.param_shapes}")
    
    def load_full_netcdf_file(self,path):
        return Dataset(path) 
    
    def get_first_valid_date_idx(self):
        for i,date in enumerate(self.dates):
            if self.does_date_exist_in_all_params(date,i):
                return i
        print("No valid date was found (no date that exists in all parameters). Aborting.")
        return None
    
    def print_param_info(self,param):
        good_idx = self.get_first_valid_date_idx()
        if good_idx is None:
            print("Please retry setting idxs")
            return
        print("Param's paths look like:",self.paths[param][good_idx])
        file = self.load_full_netcdf_file(self.paths[param][good_idx])
        print("Full file keys:", file.variables.keys())
        print("Param:",param)
        print("Param info:")
        param_data = file[param]
        print(param_data)
        print("Indices values (value  [idx]):")
        for key in self.netcdf_keys[param]:
            print("\nDim: ", key)
            values = file[key][:].data
            for i,v in enumerate(values):
                print(f"      {v} [{i}]")
        print(f"\nWhen choosing indices (using self.set_idxs()), the order of indices should be {[key for key in self.netcdf_keys[param]]} (as np.arrays). Mind the order of the indices")

    def set_idxs(self, param, idxs_list):
        # idxs_list is a list of np.arrays
        size_result = 1
        self.param_idxs[param] = idxs_list
        good_idx = self.get_first_valid_date_idx()
        if good_idx is None:
            print("Please retry setting idxs")
            return
        print("The values of each key are set to:")
        data = self.load_full_netcdf_file(self.paths[param][good_idx])
        for i,key in enumerate(self.netcdf_keys[param]):
            values = data[key][idxs_list[i]]
            print("*** Key:", key)
            size_value = values.size
            print("Size: ", size_value)
            print(values)
            size_result *= size_value
        print("\nResulting param size will be ", size_result)
   
    def is_small_res(self, param):
        # Assuming a small resolution shape is lower than [361, 720]:
        shape = self.param_shapes[param]
        if shape[-1]<720 and shape[-2]<361:
            return True
        return False
    
    def upsample_param(self,original):
        upsampled = scipy.ndimage.zoom(original[0], 2, order=3)
        upsampled = Image.fromarray(upsampled).resize([self.upsample_to[-1],self.upsample_to[-2]])
        upsampled = np.array(upsampled)
        return np.expand_dims(upsampled,0)
    
    def get_yearly_dataframe(self, year):
        # Assuming one date for all paramters in a row - if not: add NaN and skip date
        dates = []
        print(f"...Creating a dataframe for year {year}...")
        for date in self.dates:
            if date.year==year:
                dates+=[date]
        dataframe = pd.DataFrame() 
        for date in tqdm(dates):
            dataframe_row = pd.DataFrame({},index=[date])
            paths_and_params_from_date = self.params_to_take_from_paths[date]
            if paths_and_params_from_date == {}:
                for p in self.params:
                    dataframe_row[p] = [np.nan]
                dataframe = dataframe.append(dataframe_row)
                continue
            for path in paths_and_params_from_date.keys():
                full_file = self.load_full_netcdf_file(path)
                params = paths_and_params_from_date[path]
                for param in params:
                    param_numpy = full_file[param][self.param_idxs[param]]
                    if len(param_numpy.shape)==4: # reduce 1,c,h,w to c,h,w - assuming that is the only option
                        param_numpy = param_numpy[0]
                    if self.is_small_res(param) and self.upsample_to is not None:
                        param_numpy = self.upsample_param(param_numpy)
                    param_numpy = np.array(self.interpolate_na(param_numpy),dtype=object) # still a wierd message
                    dataframe_row[param] = [param_numpy]
            dataframe = dataframe.append(dataframe_row)
        if not self.keep_na:
            dataframe = dataframe.dropna(how="any")
        return dataframe
    
    def load_and_save_data_of_one_year(self, year, path_to_dir, base_filename):
        yearly_dataframe = self.get_yearly_dataframe(year)
        filename = path_to_dir+"/"+base_filename+"_"+str(year)+".pkl"
        torch.save(yearly_dataframe, filename)
        print(f"Saved df of year {year} to {filename}, length: {len(yearly_dataframe)}")

    def load_and_save_yearly_data(self, path_to_dir, base_filename, years=None, njobs=4): 
        wanted_years = years or list(set([d.year for d in self.dates]))
        Parallel(n_jobs=njobs,verbose=100)(delayed(
            self.load_and_save_data_of_one_year)(year, path_to_dir, base_filename) for year in wanted_years
        )

    def interpolate_na(self, param_with_na):
        # Implented based on https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
        # Assuming shape: levs=shape[0](=channels),lats=shape[1](=h),lons=shape[2](=w)
        y = np.arange(0, param_with_na.shape[1])
        x = np.arange(0, param_with_na.shape[2])
        grid_x, grid_y = np.meshgrid(x, y)
        param_interpolated = np.zeros_like(param_with_na)
        for c in range(param_with_na.shape[0]):
            channel_masked_invalid = np.ma.masked_invalid(param_with_na[c,:,:])
            if channel_masked_invalid[channel_masked_invalid.mask].size == 0:
                param_interpolated[c,:,:]=param_with_na[c,:,:]
                continue            
            x_valid_only = grid_x[~channel_masked_invalid.mask]
            y_valid_only = grid_y[~channel_masked_invalid.mask]
            channel_valid_only = channel_masked_invalid[~channel_masked_invalid.mask]
            param_interpolated[c,:,:] = interpolate.griddata((x_valid_only, y_valid_only), 
                                                             channel_valid_only.ravel(),
                                                             (grid_x, grid_y),method='cubic')
        return param_interpolated
    
    def save_dataframe(self, dataframe, filename):
        if filename[-4:] == ".pkl":
            dataframe.to_pickle(filename)
            print(f"Saved dataframe to file {filename}")
        else:
            print("Could not save - bad file name. Has to end with .pkl.")

    def save_dataframe_torch(self, dataframe, filename):
        if filename[-4:] == ".pkl":
            torch.save(dataframe,filename)
            print(f"Saved dataframe to file {filename}")
        else:
            print("Could not save - bad file name. Has to end with .pkl.")

    def save_dataframe_wanted_year(self, dataframe, filename, year):
        df_by_year = dataframe[dataframe.index.year==year]
        torch.save(df_by_year, filename)
#         df_by_year.to_pickle(filename)
        print(f"Saved dataframe of year {year} to file {filename}, length: {len(df_by_year)}")

    def get_years_list(self):
        return list(set([date.year for date in self.dates])) 
    
    def save_dataframe_by_years(self, dataframe, path_to_dir, base_filename):
        years_list = self.get_years_list()
        for y in years_list:
            self.save_dataframe_wanted_year(dataframe, path_to_dir+"/"+base_filename+"_"+str(y)+".pkl", y)
            
            
## Saving with torch:            
# #import pandas as pd
# date1=meteo_handler.dates[0]
# date2=meteo_handler.dates[1]

# df = pd.DataFrame({},index=[date1])
# np_obj = np.array(np.arange(10), dtype=object)
# np_float = np_obj.astype(float)
# np_float
# df["Z"] = [np_obj]
# print(df)
# torch.save(df,"test.pkl")
# loaded = torch.load("test.pkl")
# torch.tensor(df["Z"][0].astype(float))
        
# from netCDF4 import Dataset
# file = Dataset("/work/meteogroup/cams/2003/12/D20031214_12")
# file.variables
