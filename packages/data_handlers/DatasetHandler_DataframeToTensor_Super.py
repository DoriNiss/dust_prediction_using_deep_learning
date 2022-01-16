import torch
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from joblib import Parallel, delayed #conda install -c anaconda joblib

class DatasetHandler_DataframeToTensor_Super:
    
    def __init__(self, dataframe, dims_cols_strings, metadata={}, timestamps=None, save_base_filename=None,
                 invalid_col_fill=-777):
        """
        dataframe - already uploaded, to be converted into a tensor
        timestamps - if None, will use all (unless built yearly)
        data_suffixes - the suffix of data cols, e.g. " 
            "AFULA_PM10_0","BEER_SHEVA_PM25_delta_m24","BEER_SHEVA_values_count_48"
        dims_cols_strings - dim 0 is always the dataframe's index at timestamps. 
            The dims' order is used to create the dataframe's columns.
            i.e. the first columns is f"{dims_cols_strings[0][0]}_{dims_cols_strings[1][0]}_...
            and will be postitioned in the resulting tensor's [0,0,...]
            {1:[str1,str2,...],2:[str1,str2,...],...}
            e.g.: {1:["station1","stations2","stations3"],2:["PM10","PM25"],3:[0","delta_m24","values_count_48"]}
                    The resulting tensor's shape will be: [timestamps,3,2,3]
                    Assuming valid keys: all values from 1 to the highest dim, 3 in the example
        parallel_cpus - used for parallel creation of yearly tensors. 0 means no parallelism
        """
        self.dims_cols_strings = dims_cols_strings
        self.timestamps = timestamps or dataframe.index
        self.dataframe = dataframe
        self.metadata = metadata
        self.save_base_filename = save_base_filename
        self.invalid_col_fill = invalid_col_fill
        if timestamps is not None: self.sync_dataframe_timestamps()
        self.init_shape()
        self.create_metadata() 
        print("Done initiation, use self.create_yearly_datasets(years) to create and save yearly datasets,")
        print("or self.create_yearly_datasets_parallel(years,njobs=3) to create and save yearly datasets paralelly.")
        print("To create one tensor from all yearly created datasets, use the static method\n" \
              "load_merge_and_save_yearly_tensors_by_timestamps(base_filename,years,timestamps,metadata=None)")
        
    def sync_dataframe_timestamps(self):
        print("Syncing timestamps: ...")
        existing_new_timestamps = []
        full_length = len(self.timestamps)
        for t in tqdm(self.timestamps):
            try:
                self.dataframe.loc[t]
                existing_new_timestamps.append(t)
            except:
                continue
        self.timestamps = pd.to_datetime(existing_new_timestamps)
        self.dataframe = self.dataframe.loc[self.timestamps]
        print(f"... Done! Synced timestamps: original length: {full_length}, current length: {len(self.timestamps)}")
        
    def init_shape(self):
        num_dims = len(self.dims_cols_strings.keys())+1
        self.shape = [len(self.timestamps)]+[len(self.dims_cols_strings[i]) for i in range(1,num_dims)]
        print(f"Initiated full shape: {self.shape}")
            
    def create_metadata(self):
        """ Will be implemented specifically for each dataset type (e.g.  dust, meteorology...)"""
        return
    
    def get_tensor_from_timestamps(self, timestamps, print_prefix=""): 
        """ Will be overidden by meteorology dataset creation handler"""
        tensor = torch.zeros([len(timestamps)]+[i for i in self.shape[1:]])
        df = self.dataframe.loc[timestamps]
        print(f"{print_prefix}Creating tensor for {timestamps[0]} to {timestamps[-1]}, with the shape {tensor.shape}: ...")
        last_dim = list(self.dims_cols_strings.keys())[-1]
        def recursively_populate_tensor_from_col(t,dim,col_name_so_far):
            if dim>=last_dim:
                for col_idx,col_str in enumerate(self.dims_cols_strings[dim]):
                    col_full_name = f"{col_name_so_far}{col_str}"
                    try:
                        t[:,col_idx]+=df[col_full_name].values
                    except:
                        t[:,col_idx]+=self.invalid_col_fill
                return
            for col_idx,col_str in enumerate(self.dims_cols_strings[dim]):
                recursively_populate_tensor_from_col(t[:,col_idx],dim+1,col_name_so_far+f"{col_str}_")
        recursively_populate_tensor_from_col(tensor,1,"")
        print(f"{print_prefix}... Done!")
        return tensor
        
    def create_yearly_datasets(self, years):
        for year in years:
            timestamps = self.dataframe.index[self.dataframe.index.year==year]
            if timestamps.empty:
                print(f"Found no rows for year {year}, skipping year")
                continue
            t = self.get_tensor_from_timestamps(timestamps, print_prefix=f"{year}: ")
            if self.save_base_filename is not None:
                torch.save(timestamps,f"{self.save_base_filename}_{year}_timestamps.pkl")
                torch.save(t,f"{self.save_base_filename}_{year}_tensor.pkl")
    
    def create_yearly_datasets_parallel(self, years, njobs=3):
        Parallel(n_jobs=njobs,verbose=100)(delayed(self.create_yearly_datasets)([year]) for year in years)      
    
    @staticmethod
    def save_dataset(t, timestamps, base_filename):
        torch.save(timestamps,f"{base_filename}_timestamps.pkl")
        torch.save(t,f"{base_filename}_tensor.pkl")

    @staticmethod
    def merge_by_timestamps(tensors_list, timestamps_lists):
        timestamps = [t for t in timestamps_lists[0]]
        if len(timestamps_lists)>1:
            for timestamps_list in timestamps_lists[1:]:
                timestamps+=[t for t in timestamps_list]
        t = torch.cat(tensors_list,dim=0)
        timestamps = pd.to_datetime(timestamps)
        return t,timestamps

    @staticmethod
    def load_merge_and_save_yearly_tensors_by_timestamps(base_filename, years, metadata=None):
        """ Updates metadata's times and saves as a new tensor, with the suffix _merged.pkl"""
        print("Loading tensors: ...")
        tensors_list, timestamps_lists = [],[]
        for year in tqdm(years):
            try:
                tensors_list.append(torch.load(f"{base_filename}_{year}_tensor.pkl"))
                timestamps_lists.append(torch.load(f"{base_filename}_{year}_timestamps.pkl"))
            except:
                print(f"No files found for year {year}: {base_filename}_{year}_tensor.pkl")
        print("... Done! Merging tensors and saving: ...")
        t,timestamps = DatasetHandler_DataframeToTensor_Super.merge_by_timestamps(tensors_list, timestamps_lists)
        print(f"... Done! Sizes: {t.shape}, {len(timestamps)}")
        print(timestamps)
        DatasetHandler_DataframeToTensor_Super.save_dataset(t, timestamps, f"{base_filename}_merged")
        if metadata is not None:
            metadata["merged"] = f"Merged timestamps: from {timestamps[0]} to {timestamps[-1]}"
            torch.save(metadata,f"{base_filename}_merged_metadata.pkl")
        print(f"Saved as {base_filename}_merged_timestamps.pkl and {base_filename}_merged_tensor.pkl")

    @staticmethod
    def merge_as_channels(tensors, dim):
        # merge_metadata()...
        raise NotImplemented



