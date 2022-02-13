# +
import torch
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from joblib import Parallel, delayed #conda install -c anaconda joblib

from data_handlers.DatasetHandler_DataframeToTensor_Super import *


# -

class DatasetHandler_DataframeToTensor_Meteorology(DatasetHandler_DataframeToTensor_Super):
    
    """ 
        Assuming shape: [timestamps,channels,lons,lats]. 
        dims_cols_strings=[1:[list_of_channel_names]] 
        param_shape: shape of each channel. Assumed to be equal for all channels. I.e., [81,189]
        old_format_channels_dict: from old version of dataframes. 
            {col_str:{"tensor_channels":[...],"df_channels":[...]}}
    """
    def __init__(self, dataframe, dims_cols_strings, metadata={}, timestamps=None, save_base_filename=None,
                 invalid_col_fill=-777, param_shape=[], lons=None, lats=None, save_timestamps=False,
                 old_format_channels_dict=None,verbose=1):
        self.param_shape = param_shape
        self.old_format_channels_dict = old_format_channels_dict
        self.lons = lons
        self.lats = lats
        super().__init__(dataframe, dims_cols_strings, metadata=metadata, timestamps=timestamps, 
                         save_base_filename=save_base_filename,invalid_col_fill=invalid_col_fill,
                         save_timestamps=save_timestamps,verbose=verbose)
                
    def init_shape(self):
        self.shape = [len(self.timestamps),len(self.dims_cols_strings[1])]+self.param_shape
        if self.old_format_channels_dict is not None:
            self.shape[1] = sum(len(channels["tensor_channels"]) 
                                for channels in self.old_format_channels_dict.values())
        if self.verbose>0: print(f"Initiated full shape: {self.shape}")
            
    def create_metadata(self):
        self.metadata["dims"] = {
            "general": f"[timestamps,channels,lons,lats] = {self.shape}",
            "lons": {i: self.lons[i] for i in range(len(self.lons))},
            "lats": {i: self.lats[i] for i in range(len(self.lats))}
        }
        if self.old_format_channels_dict is not None:
            self.metadata["dims"]["old_format_channels_dict"] = self.old_format_channels_dict
        else: 
            self.metadata["dims"]["channels"]: {c: self.dims_cols_strings[1][c] for c in range(self.shape[1])}
        if self.save_base_filename is not None:
            torch.save(self.metadata,f"{self.save_base_filename}_metadata.pkl")    
    
    def get_tensor_from_timestamps(self, timestamps, print_prefix=""): 
        tensor = torch.zeros([len(timestamps)]+[i for i in self.shape[1:]])
        df = self.dataframe.loc[timestamps]
        if self.verbose>0: print(f"{print_prefix}Creating tensor for {timestamps[0]} to {timestamps[-1]}, with the shape {tensor.shape}: ...")
        for channel_idx,channel_string in enumerate(self.dims_cols_strings[1]):
            tensor_channel = channel_idx
            expected_shape = (tensor.shape[-2],tensor.shape[-1])
            x_as_list = []
            invalid_filled_np = np.ones_like((tensor[0,0,:,:]).numpy())
            for x_row in df[channel_string]:
                try:
                    if x_row<0: # Assuming all invalid vlues in dataframe turned into negative values
                        x_as_list.append(invalid_filled_np*x_row)
                except Exception as e:
                    try:
                        x_row_as_np = x_row.astype("float32")
                        if (x_row_as_np.shape[-2],x_row_as_np.shape[-1]) == expected_shape:
                            x_as_list.append(x_row_as_np)
                        else:
                            x_as_list.append(invalid_filled_np*self.invalid_col_fill) # Bad shape infill value
                    except Exception as e:
                        x_as_list.append(invalid_filled_np*self.invalid_col_fill) # Any other problem
            x = torch.tensor(x_as_list)
            if self.old_format_channels_dict is not None:
                old_format_df_channels = self.old_format_channels_dict[channel_string]["df_channels"]
                tensor_channel = self.old_format_channels_dict[channel_string]["tensor_channels"]
                x = x[:,old_format_df_channels,:,:]
            tensor[:,tensor_channel,:,:] = x
        if self.verbose>0: print(f"{print_prefix}... Done!")
        return tensor
    



