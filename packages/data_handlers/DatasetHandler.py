import torch
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

# splits to valid and train by years - no hard negative mining
# dust statistics - events/clear ratio
# builds a combined dataframe from both dust and era5 dataframes
# creates metadata - dust, meteorology
# creates big tensors - dust, meteorology + normalizations

class DatasetHandler:
    '''
        Used for:
        1. Creating a combined dataframe of both meteorolgy and dust
        2. Splits the data into 2 dataframes by wanted years - for Train/Valid split
        3. Print statistics (Events/Clear ratio...) - assuming a known event threshold (th)
        4. Creating metadata instances to describe the data
        5. Creating 4 tensors: for each train/valid set, creating dust,meteorology tensors
        How to work with this class:
        1. init with the dataframes' filenames, this will combine the dataframes and save
        the combined dataframe in self.dataframe 
        If you want to save the combined dataframe for loading it later, set filename_combined
        (it will save the whole Dataset_handler so it can be used later)
        The default values of columns are assumed to be:
        ['SLP', 'Z', 'U', 'V', 'PV', 'dust_0', 'delta_0', 'dust_m24',
       'delta_m24', 'dust_24', 'delta_24', 'dust_48', 'delta_48', 'dust_72',
       'delta_72']
        2. .split_train_valid with the wanted years' list
        3. .create_datasets - this will build the tensors and metadata and return a dict:
            {data_train_meteorology: train_meteorology_tensor, data_train_dust:..., 
            data_valid_meteorology:..., data_valid_dust:...,
            metadata_train_meteorology: metadata_object, metadata_train_dust:..., 
            metadata_valid_meteorology:..., metadata_valid_dust:...}
        If you already have training and validation dataframes, you can load them with
        self.load_train_valid_df(filename_train_df, filename_valid_df) and continue
        to .create_datasets (step 3)
        if keep_na==True: will add all missing datetimes in a new column, NaN values will be added to missing rows
    '''
    def __init__(self,filename_meteorology, filename_dust, th=73.4, filename_combined=None,
                 meteorology_columns_idxs=[0,1,2,3,4], dust_columns_idxs=[5,6,7,8,9,10,11,12,13,14],
                 keep_na=False):
        self.filename_meteorology = filename_meteorology
        self.filename_dust = filename_dust
        self.th = th
        self.meteorology_columns_idxs = meteorology_columns_idxs
        self.dust_columns_idxs = dust_columns_idxs
        self.combine_dataframes()
        self.keep_na = keep_na
        if filename_combined is not None:
            torch.save(self, filename_combined)
            print(f"Handler saved to {filename_combined}")

    def combine_dataframes(self):
        print("Loading meteorology dataframe: ...")
        df_meteorology = pd.read_pickle(self.filename_meteorology)
        print("... Done!")
        # print(df_meteorology.describe()) # veeeery slow
        print("Loading dust dataframe: ...")
        df_dust = pd.read_pickle(self.filename_dust)
        print("... Done!")
        # print(df_dust.describe())        
        print("Combining datasets: ...")
        self.dataframe = df_meteorology.join(df_dust, how="inner")
        print("... Done!")
        # print(self.dataframe.describe())      
        print("Removing NaN values: ...")  
        self.dataframe.dropna(how="any")
        print("... Done! The resulting dataframe is saved in self.dataframe:")
        self.print_statistics(self.dataframe)

    def print_statistics(self, df):
        print("Number of values per column:")
        print(df.count())
        print("First rows:")
        print(df.head())
        print("Last rows:")
        print(df.tail())
        num_events = df[df["dust_0"]>=self.th]["dust_0"].count()
        num_clear = df[df["dust_0"]<self.th]["dust_0"].count()
        print(f"\nNum events: {num_events}, num clear: {num_clear}, events/clear ratio: " \
            f"{100.0*num_events/num_clear:.2f}%")

    def split_train_valid(self, train_years, valid_years):
        print(f"Building training dataframe for years {train_years}: ...")
        yearly_df_list = [self.dataframe[self.dataframe.index.year==year] for year in train_years]
        self.train_df = pd.concat(yearly_df_list)
        print(f"... Done! saved in self.train_df, Building validation dataframe for years {valid_years}: ...")
        yearly_df_list = [self.dataframe[self.dataframe.index.year==year] for year in valid_years]
        self.valid_df = pd.concat(yearly_df_list)
        print("... Done! saved in self.valid_df")
        print("Train dataframe:")
        self.print_statistics(self.train_df)
        print("Valid dataframe:")
        self.print_statistics(self.valid_df)

    # self.load_train_valid_df()
    
    def create_datasets(self,folder_path=""):
        tensor_train_meteorology, tensor_train_dust, tensor_valid_meteorology, tensor_valid_dust = self.create_tensors(folder_path=folder_path)
        # self.create_metadata(folder_path=folder_path)
        out_tensors = {
            "data_train_meteorology": tensor_train_meteorology, 
            "data_train_dust": tensor_train_dust, 
            "data_valid_meteorology": tensor_valid_meteorology, 
            "data_valid_dust": tensor_valid_dust
        }
        print("\nResulting dataset:")
        for key in out_tensors.keys():
            print(f"{key}: {out_tensors[key].shape}")
        return out_tensors

    def create_tensors(self,folder_path=""):
        print("Creating training meteorology tensor: ...")
        tensor_train_meteorology = self.create_meteorology_tensor(self.train_df)
        # metadata_train_meteorology = self.create_meteorology_metadata(self.train_df)
        print(f"... Done! Resulting tensor shape: {tensor_train_meteorology.shape}")
        print("Creating training dust tensor: ...")
        tensor_train_dust = self.create_dust_tensor(self.train_df)
        print(f"... Done! Resulting tensor shape: {tensor_train_dust.shape}")
        print("Creating validation meteorology tensor: ...")
        tensor_valid_meteorology = self.create_meteorology_tensor(self.valid_df)
        print(f"... Done! Resulting tensor shape: {tensor_valid_meteorology.shape}")
        print("Creating validation dust tensor: ...")
        tensor_valid_dust = self.create_dust_tensor(self.valid_df)
        print(f"... Done! Resulting tensor shape: {tensor_valid_dust.shape}")
        if folder_path!="":
            torch.save(tensor_train_meteorology, folder_path+"tensor_train_meteorology.pkl")
            print("Saved tensor_train_meteorology.pkl")
            torch.save(tensor_train_dust, folder_path+"tensor_train_dust.pkl")
            print("Saved tensor_train_dust.pkl")
            torch.save(tensor_valid_meteorology, folder_path+"tensor_valid_meteorology.pkl")
            print("Saved tensor_valid_meteorology.pkl")
            torch.save(tensor_valid_dust, folder_path+"tensor_valid_dust.pkl")
            print("Saved tensor_valid_dust.pkl")
        return tensor_train_meteorology, tensor_train_dust, tensor_valid_meteorology, tensor_valid_dust

    def create_meteorology_tensor(self, dataframe):
        """
            Assuming all columns have the same W,H and only differ in C, 
            where shape=[1,C,H,W] or =[1,H,W] (assuming the later if len(shape)==3)
            For metadata: the default order of idxs (of the resulting channels C, [:,C,:,:]):
            [0]:'SLP', 
            [1,2,3]:'Z'[850. 500. 250.], 
            [4,5,6]:'U'[850. 500. 250.], 
            [7,8,9]:'V'[850. 500. 250.], 
            [10,11,12,13,14,15,16]:'PV'[310. 315. 320. 325. 330. 335. 340.]
        """
        list_for_stacking_per_date = []
        params = [dataframe.columns[col_name_idx] for col_name_idx in self.meteorology_columns_idxs]
        for date in tqdm(dataframe.index):
            list_for_stacking_per_channel = []
            for param in params:
                param_array = dataframe[param][date]
                if len(param_array.shape)==3:
                    list_for_stacking_per_channel.append(np.expand_dims(param_array,0))
                else:
                    channels_list = np.split(param_array,param_array.shape[1],axis=1)
                    list_for_stacking_per_channel += channels_list
            date_full_np_array = np.stack(list_for_stacking_per_channel,axis=2).squeeze(0)
            list_for_stacking_per_date.append(date_full_np_array)
        return torch.tensor(np.stack(list_for_stacking_per_date,axis=1).squeeze(0))
    
    def create_dust_tensor(self, dataframe):
        """
            'dust_0', 'delta_0', 'dust_m24',
            'delta_m24', 'dust_24', 'delta_24', 'dust_48', 'delta_48', 'dust_72',
            'delta_72']
        """
        dust_columns = [dataframe.columns[col_idx] for col_idx in self.dust_columns_idxs]
        dust_np_list_per_lag = []
        for column in dust_columns:
            dust_np_list_per_lag.append(dataframe[column].values) 
        return torch.tensor(np.stack(dust_np_list_per_lag, axis=1))
    
    def create_metadata(self,folder_path=""):
        print("To be implemented...")
        return
