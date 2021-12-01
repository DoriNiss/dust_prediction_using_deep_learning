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
        
    def translate_original_idx_to_handler(self, original_idx): # Terribly slow
        """
            e.g.: Returns 0 for original_idx=24, if the rows_that_have_sequences[0]==24 
            If the wanted idx does not appear in rows_that_have_sequences, returns None
        """
        if original_idx not in self.rows_that_have_sequences:
            return None
        return self.rows_that_have_sequences.index(original_idx)
    
    def translate_handler_idx_to_original(self, handler_idx):
        """
            e.g.: Returns 24 for handler_idx=0, if the rows_that_have_sequences[0]==24 
        """
        return self.rows_that_have_sequences[handler_idx]

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

    def get_batched_sequences(self,x,batch_idxs,add_dim=True):
        """
            Iterates through all rows in batch_idxs and returns a new tensor of all sequences in batch_idxs, 
            of shape [self.num_sequences]+self.get_sequence(x,0,add_dim=add_dim).shape
            Warning: Might be highly memory expensive
        """
        return torch.stack([self.get_sequence(x,i,add_dim=add_dim) for i in batch_idxs])
    
    def get_batched_sequences_from_original_idxs(self,x,original_idxs,add_dim=True):
        """
            Same as get_batched_sequences, but first translates the wanted original idxs to the corresponding
            idxs of the handler. Useful for contructing sequences of known label, e.g. pulling events' sequences
        """
        idxs = [self.translate_original_idx_to_handler(i) for i in original_idxs]
        idxs = [i for i in idxs if i is not None]
        return self.get_batched_sequences(x,idxs,add_dim=add_dim)

    def get_dataset_from_handler_idx(self,handler_idx,inputs_to_sequence,targets,timestamps,add_dim=True):
        """
            All inputs should have the same length at dim 0
            Returns sequenced input, target and timestamp
        """
        input_sequenced = self.get_sequence(inputs_to_sequence,handler_idx,add_dim=add_dim)
        original_idx = self.translate_handler_idx_to_original(handler_idx)
        return input_sequenced,targets[original_idx],timestamps[original_idx]

    def get_dataset_from_original_idx(self,original_idx,inputs_to_sequence,targets,timestamps,add_dim=True):
        """
            All inputs should have the same length at dim 0
            Returns sequenced input, target and timestamp
        """
        handler_idx = self.translate_original_idx_to_handler(original_idx)
        if handler_idx is None:
            print("Original row does not have a sequence, returning None's")
            return None,None,None
        input_sequenced = self.get_sequence(inputs_to_sequence,handler_idx,add_dim=add_dim)
        return input_sequenced,targets[original_idx],timestamps[original_idx]
    
    def get_batched_dataset_from_handler_idxs(self,handler_idxs,inputs_to_sequence,targets,timestamps,add_dim=True):
        inputs_sequenced = self.get_batched_sequences(inputs_to_sequence,handler_idxs,add_dim=add_dim)  
        original_idxs = [self.translate_handler_idx_to_original(handler_i) for handler_i in handler_idxs]
        return inputs_sequenced,targets[original_idxs],timestamps[original_idxs]

    def get_batched_dataset_from_original_idxs(self,original_idxs,inputs_to_sequence,targets,timestamps,add_dim=True):
        inputs_sequenced = self.get_batched_sequences_from_original_idxs(inputs_to_sequence,original_idxs,add_dim=add_dim)   
        return inputs_sequenced,targets[original_idxs],timestamps[original_idxs]
    
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

