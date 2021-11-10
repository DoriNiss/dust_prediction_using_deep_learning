import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

class SequentialHandler:
    
    def __init__(self,timestamps,sequence_items_idxs,timesteps=[6,"h"],verbose_level=1):
        print("Initializing sequences...")
        self.timestamps = timestamps
        self.sequence_items_idxs = sequence_items_idxs
        self.timesteps = timesteps
        self.verbose_level = verbose_level
        self.len_sequence = len(sequence_items_idxs)
        self.init_sequences()
        self.num_sequences = len(self.rows_that_have_sequences)
        print("...Done!")
        if self.verbose_level>0: self.print_sample(0)
        
    def init_sequences(self):
        timestamps_df = pd.DataFrame({"times":self.timestamps})
        sequence_items = [pd.Timedelta(i*self.timesteps[0],unit=self.timesteps[1]) 
                          for i in self.sequence_items_idxs]
        len_seq = self.len_sequence
        rows_that_have_sequences,sequential_rows_idxs = [],[]
        for t_idx,t in enumerate(self.timestamps):
            wanted_times = [t+seq_item for seq_item in sequence_items]
            t_sequence_idxs = []
            for wanted_time in wanted_times:
                wanted_time_idx = timestamps_df[timestamps_df["times"]==wanted_time].index.tolist()
                if wanted_time_idx==[]:
                    t_sequence_idxs = []
                    break
                t_sequence_idxs+=wanted_time_idx
            if len(t_sequence_idxs) != len_seq:
                if self.verbose_level>1: print(f"No sequence for {t}")
                continue
            rows_that_have_sequences+=[t_idx]
            sequential_rows_idxs+=[t_sequence_idxs]
        self.rows_that_have_sequences = rows_that_have_sequences
        self.sequential_rows_idxs = sequential_rows_idxs
           
    def print_sample(self,idx):
        print(f"Resulting number of sequences: {self.num_sequences}")
        print(f"Sequence items idxs: {self.sequence_items_idxs} (len={self.len_sequence}), " \
              f"for timesteps of {self.timesteps[0]}{self.timesteps[1]}, i.e.: " \
              f"{[str(i*self.timesteps[0])+self.timesteps[1] for i in self.sequence_items_idxs]}")
        print(f"Sample results:")
        print(f"   Time: {self.timestamps[self.rows_that_have_sequences[idx]]}")
        print(f"   Sequence: {[self.timestamps[i] for i in self.sequential_rows_idxs[idx]]}")
        
    def get_sequence(self,x,idx,add_dim=True):
        """
            Adds new dimension by default at position [0], e.g.: returns [seq_len,20,81,189] from [20,81,189] 
            If add_dim==False: stacking the result, e.g.: returns [seq_len*20,81,189] from [20,81,189]
        """
        seq_as_list_of_tensors = [x[i] for i in self.sequential_rows_idxs[idx]]
        seq = torch.stack(seq_as_list_of_tensors)
        if not add_dim:
            seq = seq.flatten(start_dim=0,end_dim=1)
        return seq
    
    def get_all_sequences(self,x,add_dim=True):
        """
            Iterates through all rows and returns a new tensor of all sequences, 
            of shape [self.num_sequences]+self.get_sequence(x,0,add_dim=add_dim).shape
            Warning: Might be highly memory expensive
        """
        return torch.stack([self.get_sequence(x,i,add_dim=add_dim) for i in range(self.num_sequences)])

    @staticmethod
    def get_paths_from_years_list(years_list, paths_dir, base_filename, suffix):
        """
            Assuming paths of shape: <paths_dir>/<base_filename>_<year>_<suffix>
            e.g: paths_dir="../../data/sample_data", base_filename="sample_datasets", suffix="target.pkl"
            and years_list = [2003,2004], will result in: [../../data/sample_data/sample_datasets_2003_target.pkl,
            ../../data/sample_data/sample_datasets_2004_target.pkl]
        """
        return [f"{paths_dir}/{base_filename}_{y}_{suffix}.pkl" for y in years_list]

    @staticmethod
    def get_stacked_tensor_from_paths(paths):
        """
            Assuming tesnors of shape [N,?] where N is number of datapoints and ? is the shape of each datapoint,
            e.g. [1460,20,81,189]
        """
        tensor_as_list = []
        print("Loading tensors...")
        for path in tqdm(paths):
            tensor_as_list+=[torch.load(path)]
        print("...Done! Stacking to one tensor...")
        t = torch.cat(tensor_as_list)
        print("...Done! Result shape:", t.shape)
        return t

    @staticmethod
    def get_handler_from_timestamps_paths(timestamps_paths,sequence_items_idxs,timesteps=[6,"h"],verbose_level=1):
        if verbose_level>1:
            print("Loading timestamps...")
        timestamps_list = []
        for p in timestamps_paths:
            timestamps_list+=[t for t in torch.load(p)]
        timestamps = pd.to_datetime(timestamps_list)
        if verbose_level>1:
            print(f"...Done! Number of timestamps: {len(timestamps)}, constructing handler...")
        return SequentialHandler(timestamps,sequence_items_idxs,timesteps=timesteps,verbose_level=verbose_level)
