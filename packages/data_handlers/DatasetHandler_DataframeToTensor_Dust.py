# +
import torch
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from joblib import Parallel, delayed #conda install -c anaconda joblib

from data_handlers.DatasetHandler_DataframeToTensor_Super import *

# -

class DatasetHandler_DataframeToTensor_Dust(DatasetHandler_DataframeToTensor_Super):
    
    def __init__(self, dataframe, dims_cols_strings, metadata={}, timestamps=None, save_base_filename=None,
                 invalid_col_fill=-777):
        super().__init__(dataframe, dims_cols_strings, metadata=metadata, timestamps=timestamps, 
                         save_base_filename=save_base_filename,invalid_col_fill=invalid_col_fill)
                   
    def create_metadata(self):
        def suffix_to_description(suffix_str):
            try:
                lag_str=suffix_str[suffix_str.rindex("_")+1:]
            except:
                lag_str=suffix_str
            if lag_str=="0":
                lag_description="T"
            elif lag_str[0]=="m":
                lag_description = f"T-{lag_str[1:]}h"
            else:
                lag_description = f"T+{lag_str}h"
            description_str = "Dust" if lag_str==suffix_str else suffix_str[:suffix_str.index(lag_str)-1]
            description = f"{description_str} at {lag_description}"
            return description
        def is_suffix(col_str):
            try:
                suffix=col_str[col_str.rindex("_")+1:]
                if suffix[0]=="m":
                    suffix = suffix[1:]
                successfully_parsed_to_float = float(suffix)
                return True
            except:
                return False
        self.metadata["idxs"] = {}
        self.metadata["idxs"]["dims"] = {"general":self.dims_cols_strings}
        self.metadata["idxs"]["dims"][0]=\
            f"timestamps, {self.timestamps[0]} to {self.timestamps[-1]}, len={len(self.timestamps)}"
        for dim,strings in self.dims_cols_strings.items():
            if is_suffix(strings[0]):
                idxs_dict={i: {"col string":string, "description":suffix_to_description(string)}
                           for i,string in enumerate(strings)}
            else:
                idxs_dict = {i: string for i,string in enumerate(strings)}
            self.metadata["idxs"]["dims"][dim]=idxs_dict
        if self.save_base_filename is not None:
            torch.save(self.metadata,f"{self.save_base_filename}_metadata.pkl")    




