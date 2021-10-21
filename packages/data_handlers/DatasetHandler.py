import torch
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from joblib import Parallel, delayed #conda install -c anaconda joblib

class DatasetHandler:
    
    def __init__(self, dataframes, config_dict):
        """
        cols_input, cols_target, dataframes_descriptions, 
        keep_na=False, include_all_timestamps_between=None, all_timestamps_intervals=None,
        cols_channels_input=None, cols_channels_target=None, as_float32=True):

        dataframes - list of dataframes, supposed to work per year. Each dataframe contains np.arrays with dtype = object
            The shapes of np.arrays are assumed to be [c,h,w] for meteorology dats or [1] for dust, and assuming
            target or input can be only dust or only meteorology
        dataframes_descriptions - a must have list of description per dataframe (please describe the channels). 
            e.g.: {"input":{0:"Z500",1:"PV310"},"target":{0:"dust_0",1:"lags_0"}}
        cols_channels_input/target - dict with {cols:[idxs]}, e.g. {"PV":[0], "Z":None, "U":[1,3]}. 
            If None - will use all channels (for dust dataframes keep it None)
        include_all_timestamps_between - list of [first,last] to include all dates between them, both are pandas DateTime.
            If include_all_timestamps_between==[], will init first and last from dataframes.
        all_timestamps_intervals - the intervals of all timestamps dataframe, default to "3h"
        as_float32 - if False, will set to float64 on numpy to torch conversion. Defaults to True
        
        """
        self.dataframes = dataframes
        self.cols_input = config_dict["cols_input"]
        self.cols_target = config_dict["cols_target"]
        self.cols_to_keep = self.cols_input+self.cols_target
        self.cols_channels_input = config_dict["cols_channels_input"]
        self.cols_channels_target = config_dict["cols_channels_target"]
        self.dataframes_descriptions =  config_dict["dataframes_descriptions"]
        self.keep_na =  config_dict["keep_na"]
        self.as_float32 =  config_dict["as_float32"]
        self.wanted_year =  config_dict["wanted_year"] # can be set to None
        self.include_all_timestamps_between =  config_dict["include_all_timestamps_between"]
        self.all_timestamps_intervals = config_dict["all_timestamps_intervals"] or "3h"
        if self.include_all_timestamps_between: self.init_all_timestamps()
        self.shapes = {"input": None, "target": None}
        self.col_channels_idxs = {"input": {}, "target": {}}
        self.init_cols_channels()
        self.combine_dataframes()
        self.init_shapes_and_idxs()
    
    def init_all_timestamps(self):
        first = self.dataframes[0].index[0]
        last = self.dataframes[0].index[-1]
        if len(self.dataframes)>1:
            for df in self.dataframes[1:]:
                if df.index[0] < first: first = df.index[0]
                if df.index[-1] > last: last = df.index[-1]
        self.include_all_timestamps_between = [first, last]
        all_dates =  pd.date_range(start=first, end=last, freq=self.all_timestamps_intervals, tz="UTC")
        all_dates_df = pd.DataFrame({},index=all_dates)
        if self.wanted_year is not None:
            all_dates_df = all_dates_df[all_dates_df.index.year==self.wanted_year]
        self.dataframes.append(all_dates_df)
       
    def init_cols_channels(self):
        cols_channels_input_new = {col: None for col in self.cols_input}
        cols_channels_target_new = {col: None for col in self.cols_target}
        cols_channels_new = [cols_channels_input_new, cols_channels_target_new]
        for i,cols_channels_list in enumerate([self.cols_channels_input,self.cols_channels_target]):
            if cols_channels_list is not None:
                all_cols = self.cols_input if cols_channels_list==self.cols_channels_input else self.cols_target
                for col in cols_channels_list:
                    if col not in all_cols:
                        print(f"Error! {col} from cols_channels is not in cols list: {all_cols}. Aborting...")
                        return None
                    cols_channels_new[i][col] = cols_channels_list[col]
        self.cols_channels_input,self.cols_channels_target = cols_channels_new
   
    def combine_dataframes(self, ):
        # If include_all_timestamps - assume first and last are not None (else - ignore them)
        print("Combining dataframes...")
        self.combined_dataframe = self.dataframes[0]
        if len(self.dataframes)>1:
            for df in self.dataframes[1:]:
                self.combined_dataframe = self.combined_dataframe.join(df, how="inner")
        self.combined_dataframe = self.combined_dataframe[self.cols_to_keep]
        if not self.keep_na:
            print("Removing NaN values...")  
            self.combined_dataframe.dropna(how="any")
        print("...Done! Fixing shapes of singular data cols...")
        good_sample_idx = self.get_good_combined_idx()
        for col in self.cols_to_keep:
            if len(self.combined_dataframe[col][good_sample_idx].shape)<=1:
                self.combined_dataframe[col] = self.combined_dataframe[col].astype("object")
                self.combined_dataframe[col] = [p for p in np.expand_dims(np.array(self.combined_dataframe[col]),1)]
        print("...Done! The resulting dataframe is saved in self.dataframe:")
        self.print_statistics(self.combined_dataframe)        

    def print_statistics(self, df):
        print("Number of values per column:")
        print(df.count())
        print("First rows dates:")
        print(df.index[:5])
        print("Last rows dates:")
        print(df.index[-5:])
        
    def get_good_combined_idx(self):
        for i,date in enumerate(self.combined_dataframe.index):
            if(self.combined_dataframe[i:i+1].isna().values.any()):
                continue
            return i
        print("Did not find any row without NaN")
        return None
    
    def get_shapes_of_data(self, x):
        shapes = x.shape
        if len(shapes)<=1:
            return 1,0,0
        else:
            return shapes
    
    def init_shapes_and_idxs(self):
        """
            shapes are assumed to be [c,h,w] for meteorology data, or [1] for dust, and [h,w] are the same for 
            for all inputs and targets (separately)
        """
        good_idx = self.get_good_combined_idx() # An idx with full row in self.combined_dataframe
        for dataset_type in ["input", "target"]:
            channels_counter = 0
            cols_list = self.cols_input if dataset_type=="input" else self.cols_target
            sample_data = self.combined_dataframe[cols_list[0]][good_idx]
            _,h_all,w_all=self.get_shapes_of_data(sample_data)
            for col in cols_list:
                x = self.combined_dataframe[col][good_idx]
                c,h,w=self.get_shapes_of_data(x)
                if w!=w_all or h!=h_all:
                    print(f"Bad shapes of input parameters! {h,w} and in {col} not {h_all,w_all}. Aborting...")
                    return
                self.col_channels_idxs[dataset_type][col] = np.arange(channels_counter,channels_counter+c)
                channels_counter+=c
            self.shapes[dataset_type] = [channels_counter,h_all,w_all]
       
    def create_tensor_from_dataset_type(self, dataset_type):
        shape = self.shapes[dataset_type]
        N,C,H,W = len(self.combined_dataframe), shape[0], shape[1], shape[2]
        if H==0 and W==0:
            x = torch.zeros([N,C],dtype=torch.float32)
        else:
            x = torch.zeros([N,C,H,W],dtype=torch.float32)
        cols_list = self.cols_input if dataset_type=="input" else self.cols_target
        for col in cols_list:
            channels = self.col_channels_idxs[dataset_type][col]
            if H==0 and W==0:
                x[:,channels] = torch.tensor([c.astype("float32") for c in self.combined_dataframe[col]])
            else:
                x[:,channels,:,:] = torch.tensor([c.astype("float32") for c in self.combined_dataframe[col]])
        return x

    def create_and_save_dataset(self, dir_path, base_filename):
        """
            Creates a dataset from self.combined_dataframe and saves it
            A directroy named "metadata" has to be created inside dir_path
        """
        x_input = self.create_tensor_from_dataset_type("input")
        x_target = self.create_tensor_from_dataset_type("target")
        timestamps = self.combined_dataframe.index
        self.save_dataset(dir_path, base_filename, x_input, x_target, timestamps)
    
    def save_dataset(self, dir_path, base_filename, x_input, x_target, timestamps):
        filename_input = dir_path+"/"+base_filename+"_input.pkl"
        filename_target = dir_path+"/"+base_filename+"_target.pkl"
        filename_timestamps = dir_path+"/"+base_filename+"_timestamps.pkl"
        filename_description = dir_path+"/metadata/"+base_filename+"_descriptions.pkl"
        torch.save(x_input, filename_input)
        torch.save(x_target, filename_target)
        torch.save(timestamps, filename_timestamps)
        torch.save(self.dataframes_descriptions, filename_description)
                 
    def create_sequential_dataset(self):
        # Using a specific class which contains both full tensor and sequencing info and has a function that returns seq[idx]
        # TODO
        return
    
    @staticmethod
    def create_and_save_one_dataset_from_path(dataframes_paths, dataset_arguments, save_as):
        dataframes = [torch.load(path) for path in dataframes_paths]
        try:
            handler = DatasetHandler(dataframes, dataset_arguments)
            handler.create_and_save_dataset(save_as["dir_path"], save_as["base_filename"])
        except KeyError:
            print("...An error occured (most likely all are NaN's), skipping dataset......")
        
    @staticmethod
    def create_and_save_datasets_from_paths(dataframes_paths, datasets_arguments, save_as_list, njobs=3):
        """
            datasets_arguments is a list of dictionaries required to construct each dataset, while the dataframes are given 
            as a list of paths to be loaded when constructing and saving the dataset
            dataframes_paths: the paths to dataframes to be loaded
            datasets_arguments:
            {
                cols_input
                cols_target
                dataframes_descriptions
                keep_na
                include_all_timestamps_between
                all_timestamps_intervals
                cols_channels_input
                cols_channels_target
                as_float32
                wanted_year
            }
            save_as_list: [{"dir_path":..., "base_filename":...},]
        """
        num_datasets_to_save = len(dataframes_paths)
        Parallel(n_jobs=njobs,verbose=100)(delayed(
            DatasetHandler.create_and_save_one_dataset_from_path)(dataframes_paths[i], datasets_arguments[i], save_as_list[i]) 
                for i in range(num_datasets_to_save)
        )        
        

                 
