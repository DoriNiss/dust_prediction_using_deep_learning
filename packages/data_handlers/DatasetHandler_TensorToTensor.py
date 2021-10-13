import torch
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


# Splits a given dataframe to inputs and targets

class DatasetHandler_TensorToTensor:
    """
        creates input tensor, target tensor, timestamps list from a given 
        list of yearly dataframes. Description of channels have to be manually created
        input_cols: list of columns for construction of input tensor (e.g. ["PV"])
        output_cols: list of columns for construction of target tensor (e.g. ["Z"])
        The shape of resulting tensors will be [len,C,H,W], where len is the total length 
        of all dataframes, C is the total number of channels of all columns (e.g. for input
        cols of ["PV","Z"], and assuming PV has 7 cols and Z has 1, C=8), H,W are assumed
        to be the same for all the channels (it tests and stops of not)
        e.g.: dataframe_2000["PV"][0].shape = (1, 7, 81, 169)
        example of use:
        dataset_handler = DatasetHandler_TensorToTensor([dataframe_2000, dataframe_2001], ["PV"], ["Z"])
        x,y,timestamps,description = dataset_handler.build_dataset()
    """
    def __init__(self, dataframes, input_cols, target_cols, debug=False):
        self.dataframes = dataframes
        print("Inititalizing handler...")
        self.input_cols = input_cols
        self.target_cols = target_cols
        self.input_shape = []
        self.target_shape = []
        self.debug = debug
        print("...initializing and testing shapes...")
        if self.init_shapes(dataframes):
            print("...successfully initiated shapes: inputs:,",self.input_shape,"targets:",self.target_shape)
            print("Use self.build_dataset() to get the tensors and metadata")
        else:
            print("...Error! Check the shapes of given dataframes' data")
        self.input_channels_idxs,self.target_channels_idxs = {},{}
        self.init_channels_idxs()
    
    def init_shapes(self, dataframes):
        full_len = 0
        for i,df in enumerate(dataframes):
            if self.debug: print("...Dataframe #",i,"with the keys", df.keys(), ":")
            if i==0:
                last_HW_input = None 
                last_HW_target = None
            else: 
                last_HW_input = input_single_datapoint_col_shapes[self.input_cols[0]][-2],input_single_datapoint_col_shapes[self.input_cols[0]][-1]
                last_HW_target = target_single_datapoint_col_shapes[self.target_cols[0]][-2],target_single_datapoint_col_shapes[self.target_cols[0]][-1]
            input_single_datapoint_col_shapes, target_single_datapoint_col_shapes = {}, {}
            input_single_datapoint_C, target_single_datapoint_C = 0,0
            for col in self.input_cols:
                input_single_datapoint_col_shapes[col] = df[col][0].shape[1:]
                input_single_datapoint_C += input_single_datapoint_col_shapes[col][0]
                if self.debug:
                    print("......input shapes:")
                    print("......",col,":", input_single_datapoint_col_shapes[col])
            if self.debug: print("...testing input shapes:")
            if not self.test_col_shapes(input_single_datapoint_col_shapes, last_HW_input):
                return False         
            for col in self.target_cols:
                target_single_datapoint_col_shapes[col] = df[col][0].shape[1:]
                target_single_datapoint_C += target_single_datapoint_col_shapes[col][0]
                if self.debug: 
                    print("......target shapes:")
                    print("......",col,":", target_single_datapoint_col_shapes[col])
            if self.debug: print("...testing target shapes:")
            if not self.test_col_shapes(target_single_datapoint_col_shapes, last_HW_target):
                return False
            full_len += len(df)
        self.input_shape = [full_len,input_single_datapoint_C,
                            input_single_datapoint_col_shapes[self.input_cols[0]][1],
                            input_single_datapoint_col_shapes[self.input_cols[0]][2]]
        self.target_shape = [full_len,target_single_datapoint_C,
                             target_single_datapoint_col_shapes[self.target_cols[0]][1],
                             target_single_datapoint_col_shapes[self.target_cols[0]][2]]
        return True    

    def test_col_shapes(self, col_shapes, last_HW=None,H_idx=1,W_idx=2):
        cols = col_shapes.keys()
        first_col = col_shapes[list(cols)[0]]
        if last_HW is None:
            H,W = first_col[H_idx],first_col[W_idx]
        else:
            H,W = last_HW
        for col in cols:
            if col_shapes[col][H_idx]!=H or col_shapes[col][W_idx]!=W:
                print("Error with shapes of",cols,", terminating... (H,W shapes are supposed to be the same for all channels)")
                return False
        if self.debug: print("...Good shapes!")
        return True

    def init_channels_idxs(self):
        last_channel_idx = 0
        print("Channel indices:")
        print("   Input:")
        for col in self.input_cols:
            num_channels = self.dataframes[0][col][0].shape[1]
            self.input_channels_idxs[col] = [n for n in range(last_channel_idx,last_channel_idx+num_channels)]
            last_channel_idx += num_channels
            print(f"      {col}: {self.input_channels_idxs[col]}")
        last_channel_idx = 0
        print("   Target:")
        for col in self.target_cols:
            num_channels = self.dataframes[0][col][0].shape[1]
            self.target_channels_idxs[col] = [n for n in range(last_channel_idx,last_channel_idx+num_channels)]
            last_channel_idx += num_channels
            print(f"      {col}: {self.target_channels_idxs[col]}")       

    def build_dataset(self):
        input_tensor = torch.zeros(self.input_shape)
        target_tensor = torch.zeros(self.target_shape)
        timestamps_list = []
        print(f"Building tensors and metadata, of shapes: input{self.input_shape}, target{self.target_shape}, timestamps[{self.input_shape[0]}]...")
        for i,df in enumerate(self.dataframes):
            print(f"...building from dataframe #{i} ({i+1} out of {len(self.dataframes)})...")
            for t,timestamp in enumerate(tqdm(df.index)):
                for col in self.input_cols:
                    c_idxs = np.array(self.input_channels_idxs[col])
                    input_tensor[t,c_idxs,:,:] = torch.tensor(df[col][t][0,:,:,:])
                for col in self.target_cols:
                    c_idxs = np.array(self.target_channels_idxs[col])
                    target_tensor[t,c_idxs,:,:] = torch.tensor(df[col][t][0,:,:,:])
                timestamps_list.append(timestamp)
        timestamps = pd.to_datetime(timestamps_list)
        print(f"...Done! The tensors are from time {timestamps[0]} to {timestamps[-1]}")
        return input_tensor, target_tensor, timestamps
    