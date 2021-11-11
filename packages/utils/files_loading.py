import torch
from tqdm import tqdm
import pandas as pd

def get_paths_from_years_list(years_list, paths_dir, base_filename, suffix):
    """
        Assuming paths of shape: <paths_dir>/<base_filename>_<year>_<suffix>
        e.g: paths_dir="../../data/sample_data", base_filename="sample_datasets", suffix="target"
        and years_list = [2003,2004], will result in: [../../data/sample_data/sample_datasets_2003_target.pkl,
        ../../data/sample_data/sample_datasets_2004_target.pkl]
    """
    return [f"{paths_dir}/{base_filename}_{y}_{suffix}.pkl" for y in years_list]

# +
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

def get_stacked_timestamps_from_paths(paths):
    stacked_timestamps_datetimeindex = []
    timestamps_list = [torch.load(path) for path in paths]
    for timestamps in timestamps_list:
        stacked_timestamps_datetimeindex+=[timestamp for timestamp in timestamps]
    return pd.to_datetime(stacked_timestamps_datetimeindex)

def load_stacked_inputs_targets_timestamps_from_years_list(years_list, path_dir, base_filename):
    targets_paths = get_paths_from_years_list(years_list, path_dir, base_filename, "target")
    inputs_paths = get_paths_from_years_list(years_list, path_dir, base_filename, "input")
    timestamps_paths = get_paths_from_years_list(years_list, path_dir, base_filename, "timestamps")
    print("Creating one tensor from all targets...")
    targets = get_stacked_tensor_from_paths(targets_paths)
    print("Creating one tensor from all inputs...")
    inputs = get_stacked_tensor_from_paths(inputs_paths)
    timestamps = get_stacked_timestamps_from_paths(timestamps_paths)
    print(f"\n\nDone! Result sizes: inputs: {inputs.shape}, targets: {targets.shape}, timestamps: {len(timestamps)}")
    return inputs, targets, timestamps
    
    
