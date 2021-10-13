import glob
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch

class Meteorology_to_pandas_handler:
    def __init__(self, params=None, folders=None, prefixes=None, netcdf_keys=None,
        dates=None, debug=False):
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
        self.params = params if params is not None else self.set_default_params()
        self.folders = folders if folders is not None else self.set_default_folders()
        self.prefixes = prefixes if prefixes is not None else self.set_default_prefixes()
        self.netcdf_keys = netcdf_keys if netcdf_keys is not None else self.set_default_netcdf_keys()
        self.debug = debug
        self.dates = dates if dates is not None else self.set_default_dates()
        self.paths = {}
        self.param_idxs = {}
        self.init_paths()

    def set_default_params(self):
        return ["SLP", "Z", "U", "V", "PV"]

    def set_default_folders(self):
        return {
            "SLP":"/work/meteogroup/era5/",
            "Z":"/work/meteogroup/era5/plev/",
            "U":"/work/meteogroup/era5/plev/",
            "V":"/work/meteogroup/era5/plev/",
            "PV":"/work/meteogroup/era5/isn/",
        }

    def set_default_prefixes(self):
        return {"PV":"P","Z":"P","U":"P","V":"P","SLP":"P"}

    def set_default_netcdf_keys(self):
        return {
            "SLP":["time", "lat", "lon"],
            "Z":["time", "lev", "lat", "lon"],
            "U":["time", "lev", "lat", "lon"],
            "V":["time", "lev", "lat", "lon"],
            "PV":["time", "lev", "lat", "lon"]
        }

    def set_default_dates(self):
        if self.debug:
            return pd.date_range(start="2000-06-30 00:00", end="2000-07-05 21:00", 
                tz="UTC", freq="3h")            
        return pd.date_range(start="2000-01-01 00:00", end="2021-06-30 21:00", 
            tz="UTC", freq="3h")
   
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

    def load_full_netcdf_file(self,path):
        return Dataset(path) 

    def print_param_info(self,param):
        print("Param's paths look like:",self.paths[param][0])
        file = self.load_full_netcdf_file(self.paths[param][0])
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
        print(f"\nWhen choosing indices (using self.set_idxs()), the order of indices should be {[key for key in self.netcdf_keys[param]]} (as np.arrays)")

    def set_idxs(self, param, idxs_list):
        # idxs_list is a list of np.arrays
        size_result = 1
        self.param_idxs[param] = idxs_list
        print("The values of each key are set to:")
        data = self.load_full_netcdf_file(self.paths[param][0])
        for i,key in enumerate(self.netcdf_keys[param]):
            values = data[key][idxs_list[i]]
            print("*** Key:", key)
            size_value = values.size
            print("Size: ", size_value)
            print(values)
            size_result *= size_value
        print("\nResulting param size will be ", size_result)

    def load_data(self): # can be more effecient if each file was loaded only once
        dataframe = pd.DataFrame({},index=self.dates)
        for param in self.params:
            print(f"Loading data of {param}:")
            numpy_list = []
            for path in tqdm(self.paths[param]):
                if path=="":
                    numpy_list.append(np.nan)
                    continue
                full_file = self.load_full_netcdf_file(path)
                param_numpy = full_file[param][self.param_idxs[param]]
                # it is a masked array - so let's fix it with 0, assuming there are not too many
                param_numpy = np.ma.fix_invalid(param_numpy.data, copy=True, mask=param_numpy.mask, 
                fill_value=0).data
                numpy_list.append(param_numpy) 
            dataframe[param] = numpy_list # Raises a message about np.arrays of different sizes and prefers it as objects. Ignoring for now
        return dataframe.dropna(how="any")

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
        print(f"Saved dataframe of year {year} to file {filename}")

    def get_years_list(self):
        return list(set([date.year for date in self.dates])) 
    
    def save_dataframe_by_years(self, dataframe, path_to_dir, base_filename):
        years_list = self.get_years_list()
        for y in years_list:
            self.save_dataframe_wanted_year(dataframe, path_to_dir+"/"+base_filename+"_"+str(y)+".pkl", y)
    