#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DatasetHandler import *
from utils.files_loading import *
from data_handlers.tensors_labeling_and_averaging import *
from tqdm import tqdm


years = [y for y in range(2003, 2019)]
dataframes_dir = "../../data/meteorology_dataframes_20_81_189_3h/"


dataframes_descriptions_paths = [
    dataframes_dir+"metadata/meteorology_dataframes_20_81_189_3h_description.pkl",
]
dataframes_descriptions = [torch.load(path) for path in dataframes_descriptions_paths]


dust_data_path = "../../data/dust_20000101to20213006_3h_7days_future.pkl"


dataframes_paths = []
for y in years:
    dataframes_paths.append([
        dataframes_dir+"meteorology_dataframe_20_81_189_3h_"+str(y)+".pkl",
        dust_data_path
    ])

        
datasets_arguments = []
for y in years:        
    datasets_arguments.append(
        {
            "cols_input": ["SLP","Z","U","V","PV","aod550","duaod550","aermssdul","aermssdum"],
            "cols_target": [dataframes_descriptions[0]["target"][i]["df_col"]  for i in range(16)],
            "dataframes_descriptions": dataframes_descriptions,
            "keep_na": False,
            "include_all_timestamps_between": False,
            "all_timestamps_intervals": "3h",
            "cols_channels_input": None,
            "cols_channels_target": None,
            "as_float32": True,
            "wanted_year": y
        })

        
datasets_dir = "../../data/datasets_20_81_189_3h_7days_future"
save_as_list = []
for i in range(len(dataframes_paths)):
    save_as_list.append({
        "dir_path": datasets_dir,
        "base_filename": "dataset_20_81_189_3h_7days_future"+str(years[i])
    })


years = [y for y in range(2003, 2019)]
dataframes_dir = "../../data/meteorology_dataframes_20_81_189_3h/"


dataframes_descriptions_paths = [
    dataframes_dir+"metadata/meteorology_dataframes_20_81_189_3h_description.pkl",
]
dataframes_descriptions = [torch.load(path) for path in dataframes_descriptions_paths]





data_dir_original = "../../data/datasets_20_81_189"
data_dir_new = "../../data/datasets_20_81_189_averaged_dust_24h_past_only"
base_filename_original = "dataset_20_81_189"
base_filename_new = "dataset_20_81_189_averaged_dust_24h_past_only"
description_new_path = data_dir_new+"/metadata/"+base_filename_new+"_metadata.pkl"
suffix_target = "target.pkl"
suffix_input = "input.pkl"
suffix_timestamps = "timestamps.pkl"
years_list = list(range(2003,2019))


description_original = torch.load("../../data/meteorology_dataframes_23_81_189/metadata/meteorology_dataframes_20_81_189_description.pkl")
description_original["target"]


description = description_original.copy()
description["target"] = {}
description["target"][0] = "dust at i+0h"
# for i in range(1,8):
#     hours = [str(6*h)+"h" for h in range((i-1)*4,i*4)]
#     description["target"][i] = f"average of dust of {hours} (average of the following {(i-1)*24} to {i*24} hours)"
for i in range(8,15):
    hours = [str(-h)+"h" for h in range(24*(15-i),24*(14-i),-6)]
    description["target"][i-7] = f"average of dust of {hours} (average of the last {24*(15-i)} to {24*(14-i)} hours)"
description["target"][8] = "delta dust at i+0h"
# for i in range(1,8):
#     hours = [str(6*h)+"h" for h in range((i-1)*4,i*4)]
#     description["target"][i+15] = f"average of delta dust of {hours} (average of the following {(i-1)*24} to {i*24} hours)"
for i in range(8,15):
    hours = [str(-h)+"h" for h in range(24*(15-i),24*(14-i),-6)]
    description["target"][i+1] = f"average of delta dust of {hours} (average of the last {24*(15-i)} to {24*(14-i)} hours)"
description["target"][16] = "Label: threshold = 73.4[ug/m3], label=0 if dust_0<threshold, label=1 if dust_0>=threshold"
description["target"]


# cols after duplicating col 0 at [0] (dust 0) and col 57 (which turns to 58) at [58] (delta_0):
cols_to_avg_dust_0 = [np.array([0])]
valid_threshold_dust_0 = [0]
cols_to_avg_delta_0 = [np.array([58])]
valid_threshold_delta_0 = [0]

# cols_to_avg_dust_future = [np.arange(i,i+4) for i in range(1,29,4)] #ignoring 168h (29 in the col_0 duplicated tensor)
cols_to_avg_dust_past = [np.arange(i,i+4) for i in range(30,58,4)]
# cols_to_avg_delta_dust_future = [np.arange(i,i+4) for i in range(59,86,4)]
cols_to_avg_delta_dust_past = [np.arange(i,i+4) for i in range(88,116,4)]

valid_thresholds = [0.5]*7

cols_to_avg_label = [np.array([116])]
valid_thresholds_label = [0]


cols_to_average = (cols_to_avg_dust_0+cols_to_avg_dust_past+cols_to_avg_delta_0+
                   cols_to_avg_delta_dust_past+cols_to_avg_label)

valid_threshold = valid_threshold_dust_0+valid_thresholds+valid_threshold_delta_0+valid_thresholds+valid_thresholds_label


for i in range(len(cols_to_average)):
    print(f"\n{i}: cols: {cols_to_average[i]}, th: {valid_threshold[i]}")
    print(description["target"][i])


# description_original["target"]


# torch.save(description,description_new_path)
# torch.load(description_new_path)


thresholds = [73.4]
labels = [0,1]


for y in tqdm(years_list):
    print(f"\n#### Year {y}:")
    original_target = torch.load(f"{data_dir_original}/{base_filename_original}_{y}_{suffix_target}")
    original_input = torch.load(f"{data_dir_original}/{base_filename_original}_{y}_{suffix_input}")
    original_timestamps = torch.load(f"{data_dir_original}/{base_filename_original}_{y}_{suffix_timestamps}")
    new_target = duplicate_col_i(original_target,0)
    new_target = duplicate_col_i(new_target,58)
    new_target = add_labels(new_target,thresholds=thresholds,labels=labels,label_by_col=0)
    new_target,rows_to_keep = average_cols_and_drop_invalid(new_target,cols_to_average,valid_threshold,
                                                            invalid_values=[0, np.nan])
    print(f"Original length: {len(original_timestamps)}, new length: {len(rows_to_keep)}")
    new_target = new_target[rows_to_keep]
    new_input = original_input[rows_to_keep]
    new_timestamps = original_timestamps[rows_to_keep]
    print(f"Shapes: inputs: {new_input.shape}, targets: {new_target.shape}, timestamps: {len(new_timestamps)}")
    torch.save(new_target,f"{data_dir_new}/{base_filename_new}_{y}_{suffix_target}")
    torch.save(new_input,f"{data_dir_new}/{base_filename_new}_{y}_{suffix_input}")
    torch.save(new_timestamps,f"{data_dir_new}/{base_filename_new}_{y}_{suffix_timestamps}")
    


sample_year = 2015

sample_input = torch.load(f"{data_dir_new}/{base_filename_new}_{sample_year}_{suffix_input}")
sample_target = torch.load(f"{data_dir_new}/{base_filename_new}_{sample_year}_{suffix_target}")
sample_timestamps = torch.load(f"{data_dir_new}/{base_filename_new}_{sample_year}_{suffix_timestamps}")

print(sample_input.shape,sample_target.shape,len(sample_timestamps))


sample_target[5]





inputs_all,targets_all,timestamps_all = load_stacked_inputs_targets_timestamps_from_years_list(years_list,
                                                                                               data_dir_new, 
                                                                                               base_filename_new)
inputs_all.shape,targets_all.shape,len(timestamps_all)


torch.save(inputs_all,f"{data_dir_new}/{base_filename_new}_all_inputs.pkl")
torch.save(targets_all,f"{data_dir_new}/{base_filename_new}_all_targets.pkl")
torch.save(timestamps_all,f"{data_dir_new}/{base_filename_new}_all_timestamps.pkl")




