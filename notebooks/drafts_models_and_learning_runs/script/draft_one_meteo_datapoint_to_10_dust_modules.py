#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
import timm.models.vision_transformer as ViT
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate 
from tqdm import tqdm

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DustPredictionDataset import *
from training.train_model import *
from data_handlers.augmentations import *


# !pip install git+https://github.com/pvigier/perlin-numpy


# paths - to be saved in a file?

debug_dir = "../../data/tensors_debug_1/"
debug_meteorology_train_path = debug_dir+"tensor_train_meteorology.pkl"
debug_meteorology_valid_path = debug_dir+"tensor_valid_meteorology.pkl"
debug_dust_train_path = debug_dir+"tensor_train_dust.pkl"
debug_dust_valid_path = debug_dir+"tensor_valid_dust.pkl"

data_dir = "../../data/tensors_meteo20000101to20210630_dust_0_m24_24_48_72/"
split1_dir = data_dir+"split1_ordered/"
split2_dir = data_dir+"split2_extreme_ratios_in_train_avg_in_valid/"
split3_dir = data_dir+"split3_extreme_ratios_in_valid_avg_in_train/"
split4_dir = data_dir+"split4_max_num_events_in_train_avg_valid/"
split5_dir = data_dir+"split5_train_distant_years_valid_between/"
split_dirs = [split1_dir,split2_dir,split3_dir,split4_dir,split5_dir]
meteorology_train_paths = [split+"tensor_train_meteorology.pkl" for split in split_dirs]
meteorology_valid_paths = [split+"tensor_valid_meteorology.pkl" for split in split_dirs]
dust_train_paths = [split+"tensor_train_dust.pkl" for split in split_dirs]
dust_valid_paths = [split+"tensor_valid_dust.pkl" for split in split_dirs]

metadata_dir = "../../data/metadata_meteo20000101to20210630_dust_0_m24_24_48_72/"
metadata_columns_path = metadata_dir+"all_columns.pkl"
metadata_yearly_statistics_path = metadata_dir+"yearly_statistics.pkl"
metadata_all_times_path = metadata_dir+"all_times.pkl"
metadata_times_split1_train_path = metadata_dir+"split1_ordered_train_times.pkl"
metadata_times_split2_train_path = metadata_dir+"split2_extreme_ratios_in_train_avg_in_valid_train_times.pkl"
metadata_times_split3_train_path = metadata_dir+"split3_extreme_ratios_in_valid_avg_in_train_train_times.pkl"
metadata_times_split4_train_path = metadata_dir+"split4_max_num_events_in_train_avg_valid_train_times.pkl"
metadata_times_split5_train_path = metadata_dir+"split5_train_distant_years_valid_between_train_times.pkl"
metadata_times_train_paths = [metadata_times_split1_train_path,metadata_times_split2_train_path,
                              metadata_times_split3_train_path,metadata_times_split4_train_path,
                              metadata_times_split5_train_path]
metadata_times_split1_valid_path = metadata_dir+"split1_ordered_valid_times.pkl"
metadata_times_split2_valid_path = metadata_dir+"split2_extreme_ratios_in_train_avg_in_valid_valid_times.pkl"
metadata_times_split3_valid_path = metadata_dir+"split3_extreme_ratios_in_valid_avg_in_train_valid_times.pkl"
metadata_times_split4_valid_path = metadata_dir+"split4_max_num_events_in_train_avg_valid_valid_times.pkl"
metadata_times_split5_valid_path = metadata_dir+"split5_train_distant_years_valid_between_valid_times.pkl"
metadata_times_valid_paths = [metadata_times_split1_valid_path,metadata_times_split2_valid_path,
                              metadata_times_split3_valid_path,metadata_times_split4_valid_path,
                              metadata_times_split5_valid_path]


# TBD: get times of debug dataset
# import sys
# sys.path.insert(0, '../../packages/data_handlers')
# from Dataset_handler import *
# debug_dataset_handler = torch.load(debug_dir+"dummy_dataset_handler.pkl") # not working
times_debug = torch.load(metadata_times_split1_valid_path)[:48] # incorrect times - to be corrected
len(times_debug)


torch.load(metadata_columns_path)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device





# debug
train_dataset = DustPredictionDataset(torch.load(debug_meteorology_train_path),
                                      torch.load(debug_dust_train_path),times_debug)
valid_dataset = DustPredictionDataset(torch.load(debug_meteorology_valid_path),
                                      torch.load(debug_dust_valid_path),times_debug)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=valid_dataset.collate_fn)

model = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, depth=8,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
model = model.to(device)

lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

debug_best_path = debug_dir+"best_debug_model.pt"
debug_last_path = debug_dir+"last_debug_model.pt"

num_epochs = 8
(debug_train_losses,debug_valid_losses,
 debug_train_lags_loss,debug_train_delta_lags_losses,
 debug_valid_lags_loss,debug_valid_delta_lags_losses) = train_loop(model, optimizer, train_loader, valid_loader, 
                                                                   device, epochs=num_epochs, valid_every=1, 
                                                                   loss_cfg=None, sample_predictions_every=2, 
                                                                   sample_size=5, sample_cols=[0],loss_plot_end=True,
                                                                   save_best_model_dict_to=debug_best_path, 
                                                                   save_last_model_dict_to=debug_last_path)
debug_train_lags_loss.shape,debug_train_delta_lags_losses.shape,debug_valid_delta_lags_losses.shape


# checking augmentation:

perlin_augmentation = PerlinAugmentation(torch.load(debug_meteorology_train_path)[:11,:,:,:], 
                                         torch.load(debug_dust_train_path)[:11,:], debug=True)

train_dataset = DustPredictionDataset(torch.load(debug_meteorology_train_path),
                                      torch.load(debug_dust_train_path),times_debug, augmentation=perlin_augmentation)
valid_dataset = DustPredictionDataset(torch.load(debug_meteorology_valid_path),
                                      torch.load(debug_dust_valid_path),times_debug)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=valid_dataset.collate_fn)

model = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, depth=8,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
model = model.to(device)

lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

debug_best_path = debug_dir+"best_debug_model.pt"
debug_last_path = debug_dir+"last_debug_model.pt"

num_epochs = 8
(debug_train_losses,debug_valid_losses,
 debug_train_lags_loss,debug_train_delta_lags_losses,
 debug_valid_lags_loss,debug_valid_delta_lags_losses) = train_loop(model, optimizer, train_loader, valid_loader, 
                                                                   device, epochs=num_epochs, valid_every=1, 
                                                                   loss_cfg=None, sample_predictions_every=2, 
                                                                   sample_size=5, sample_cols=[0],loss_plot_end=True,
                                                                   save_best_model_dict_to=debug_best_path, 
                                                                   save_last_model_dict_to=debug_last_path)
debug_train_lags_loss.shape,debug_train_delta_lags_losses.shape,debug_valid_delta_lags_losses.shape





results_dir = "../../results_models/presentation/"
# presentation_set_e450_lr0p00001_augmentation
# presentation_set_e450_lr0p0001_augmentation
# presentation_set_e450_lr0p00001_noaugmentation
# presentation_set_e450_lr0p0001_noaugmentation
# data path
presentation_dir = data_dir+"presentation_set/"


# presentation_meteorology_train = torch.load(presentation_dir+"tensor_train_meteorology.pkl")
# presentation_meteorology_valid = torch.load(presentation_dir+"tensor_valid_meteorology.pkl")
# presentation_dust_train = torch.load(presentation_dir+"tensor_train_dust.pkl")
# presentation_dust_valid = torch.load(presentation_dir+"tensor_valid_dust.pkl")
# presentation_times_train = torch.load(presentation_dir+"times_train.pkl")
# presentation_times_valid = torch.load(presentation_dir+"times_valid.pkl")

# presentation_meteorology_augmentation = torch.load(presentation_dir+"tensor_augmentation_meteorology.pkl")
# presentation_dust_augmentation = torch.load(presentation_dir+"tensor_augmentation_dust.pkl")
# presentation_times_augmentation = torch.load(presentation_dir+"times_augmentation.pkl")


# print(presentation_meteorology_train.shape,presentation_dust_train.shape,presentation_times_train)
# print(presentation_meteorology_valid.shape,presentation_dust_valid.shape,presentation_times_valid)
# print(presentation_meteorology_augmentation.shape,presentation_dust_augmentation.shape,presentation_times_augmentation)





# Presentation set: 450 epochs: 1) lr = 0.0001 2) lr = 0.00001, 3,4) - same lr's, no augmentation
# For each: save 1) best valid model, 2) final model (overfitting model)


# Presentation set, 450 (200...) epochs, lr = 0.0001, augmentation

results_dir_specific = results_dir+"presentation_set_e450_lr0p0001_augmentation/"


best_model_path = results_dir_specific+"best_model_state.pt"
last_model_path = results_dir_specific+"last_model_state.pt"

train_losses_path = results_dir_specific+"train_losses.pkl"
train_lags_losses_path = results_dir_specific+"train_lags_losses.pkl"
train_delta_lags_losses_path = results_dir_specific+"train_delta_lags_losses.pkl"
valid_losses_path = results_dir_specific+"valid_losses.pkl"
valid_lags_losses_path = results_dir_specific+"valid_lags_losses.pkl"
valid_delta_lags_losses_path = results_dir_specific+"valid_delta_lags_losses.pkl"



perlin_augmentation = PerlinAugmentation(torch.load(presentation_dir+"tensor_augmentation_meteorology.pkl"), 
                                         torch.load(presentation_dir+"tensor_augmentation_dust.pkl"), debug=False)

train_dataset = DustPredictionDataset(torch.load(presentation_dir+"tensor_train_meteorology.pkl"),
                                      torch.load(presentation_dir+"tensor_train_dust.pkl"),
                                      torch.load(presentation_dir+"times_train.pkl"), 
                                      augmentation=perlin_augmentation)
valid_dataset = DustPredictionDataset(torch.load(presentation_dir+"tensor_valid_meteorology.pkl"),
                                      torch.load(presentation_dir+"tensor_valid_dust.pkl"),
                                      torch.load(presentation_dir+"times_valid.pkl"))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True, collate_fn=valid_dataset.collate_fn)

model = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, depth=8,
                              num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, distilled=False,
                              drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
model = model.to(device)

lr = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 200

(train_losses,valid_losses,
 train_lags_losses,train_delta_lags_losses,
 valid_lags_losses,valid_delta_lags_losses) = train_loop(model, optimizer, train_loader, valid_loader, 
                                                         device, epochs=num_epochs, valid_every=1, 
                                                         loss_cfg=None, sample_predictions_every=2, 
                                                         sample_size=5, sample_cols=[0],loss_plot_end=True,
                                                         save_best_model_dict_to=best_model_path, 
                                                         save_last_model_dict_to=last_model_path, debug=False)

torch.save(train_losses,train_losses_path)
torch.save(train_lags_losses,train_lags_losses_path)
torch.save(train_delta_lags_losses,train_delta_lags_losses_path)
torch.save(valid_losses,valid_losses_path)
torch.save(valid_lags_losses,valid_lags_losses_path)
torch.save(valid_delta_lags_losses,valid_delta_lags_losses_path)





# Presentation set, 300 epochs, lr = 0.00001, augmentation, weighted loss

results_dir_specific = results_dir+"presentation_set_e300_lr0p00001_augmentation_weighted_loss/"

loss_cfg = LossConfig(device, decaying_weights=True)


best_model_path = results_dir_specific+"best_model_state.pt"
last_model_path = results_dir_specific+"last_model_state.pt"

train_losses_path = results_dir_specific+"train_losses.pkl"
train_lags_losses_path = results_dir_specific+"train_lags_losses.pkl"
train_delta_lags_losses_path = results_dir_specific+"train_delta_lags_losses.pkl"
valid_losses_path = results_dir_specific+"valid_losses.pkl"
valid_lags_losses_path = results_dir_specific+"valid_lags_losses.pkl"
valid_delta_lags_losses_path = results_dir_specific+"valid_delta_lags_losses.pkl"



perlin_augmentation = PerlinAugmentation(torch.load(presentation_dir+"tensor_augmentation_meteorology.pkl"), 
                                         torch.load(presentation_dir+"tensor_augmentation_dust.pkl"), debug=False)

train_dataset = DustPredictionDataset(torch.load(presentation_dir+"tensor_train_meteorology.pkl"),
                                      torch.load(presentation_dir+"tensor_train_dust.pkl"),
                                      torch.load(presentation_dir+"times_train.pkl"), 
                                      augmentation=perlin_augmentation)
valid_dataset = DustPredictionDataset(torch.load(presentation_dir+"tensor_valid_meteorology.pkl"),
                                      torch.load(presentation_dir+"tensor_valid_dust.pkl"),
                                      torch.load(presentation_dir+"times_valid.pkl"))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True, collate_fn=valid_dataset.collate_fn)

model = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, depth=8,
                              num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, distilled=False,
                              drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
model = model.to(device)

lr = 0.00001

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 300

(train_losses,valid_losses,
 train_lags_losses,train_delta_lags_losses,
 valid_lags_losses,valid_delta_lags_losses) = train_loop(model, optimizer, train_loader, valid_loader, 
                                                         device, epochs=num_epochs, valid_every=1, 
                                                         loss_cfg=loss_cfg, sample_predictions_every=2, 
                                                         sample_size=5, sample_cols=[0],loss_plot_end=True,
                                                         save_best_model_dict_to=best_model_path, 
                                                         save_last_model_dict_to=last_model_path, debug=False)

torch.save(train_losses,train_losses_path)
torch.save(train_lags_losses,train_lags_losses_path)
torch.save(train_delta_lags_losses,train_delta_lags_losses_path)
torch.save(valid_losses,valid_losses_path)
torch.save(valid_lags_losses,valid_lags_losses_path)
torch.save(valid_delta_lags_losses,valid_delta_lags_losses_path)





# Presentation set, 450 epochs, lr = 0.0001, no augmentation

results_dir_specific = results_dir+"presentation_set_e450_lr0p0001_noaugmentation/"


best_model_path = results_dir_specific+"best_model_state.pt"
last_model_path = results_dir_specific+"last_model_state.pt"

train_losses_path = results_dir_specific+"train_losses.pkl"
train_lags_losses_path = results_dir_specific+"train_lags_losses.pkl"
train_delta_lags_losses_path = results_dir_specific+"train_delta_lags_losses.pkl"
valid_losses_path = results_dir_specific+"valid_losses.pkl"
valid_lags_losses_path = results_dir_specific+"valid_lags_losses.pkl"
valid_delta_lags_losses_path = results_dir_specific+"valid_delta_lags_losses.pkl"


train_dataset = DustPredictionDataset(torch.load(presentation_dir+"tensor_train_meteorology.pkl"),
                                      torch.load(presentation_dir+"tensor_train_dust.pkl"),
                                      torch.load(presentation_dir+"times_train.pkl"))
valid_dataset = DustPredictionDataset(torch.load(presentation_dir+"tensor_valid_meteorology.pkl"),
                                      torch.load(presentation_dir+"tensor_valid_dust.pkl"),
                                      torch.load(presentation_dir+"times_valid.pkl"))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True, collate_fn=valid_dataset.collate_fn)

model = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, depth=8,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
model = model.to(device)

lr = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 450

(train_losses,valid_losses,
 train_lags_losses,train_delta_lags_losses,
 valid_lags_losses,valid_delta_lags_losses) = train_loop(model, optimizer, train_loader, valid_loader, 
                                                         device, epochs=num_epochs, valid_every=1, 
                                                         loss_cfg=None, sample_predictions_every=2, 
                                                         sample_size=5, sample_cols=[0],loss_plot_end=True,
                                                         save_best_model_dict_to=best_model_path, 
                                                         save_last_model_dict_to=last_model_path, debug=False)

torch.save(train_losses,train_losses_path)
torch.save(train_lags_losses,train_lags_losses_path)
torch.save(train_delta_lags_losses,train_delta_lags_losses_path)
torch.save(valid_losses,valid_losses_path)
torch.save(valid_lags_losses,valid_lags_losses_path)
torch.save(valid_delta_lags_losses,valid_delta_lags_losses_path)





# Presentation set, 450 epochs, lr = 0.00001, no augmentation

results_dir_specific = results_dir+"presentation_set_e450_lr0p00001_noaugmentation/"


best_model_path = results_dir_specific+"best_model_state.pt"
last_model_path = results_dir_specific+"last_model_state.pt"

train_losses_path = results_dir_specific+"train_losses.pkl"
train_lags_losses_path = results_dir_specific+"train_lags_losses.pkl"
train_delta_lags_losses_path = results_dir_specific+"train_delta_lags_losses.pkl"
valid_losses_path = results_dir_specific+"valid_losses.pkl"
valid_lags_losses_path = results_dir_specific+"valid_lags_losses.pkl"
valid_delta_lags_losses_path = results_dir_specific+"valid_delta_lags_losses.pkl"


train_dataset = DustPredictionDataset(torch.load(presentation_dir+"tensor_train_meteorology.pkl"),
                                      torch.load(presentation_dir+"tensor_train_dust.pkl"),
                                      torch.load(presentation_dir+"times_train.pkl"))
valid_dataset = DustPredictionDataset(torch.load(presentation_dir+"tensor_valid_meteorology.pkl"),
                                      torch.load(presentation_dir+"tensor_valid_dust.pkl"),
                                      torch.load(presentation_dir+"times_valid.pkl"))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True, collate_fn=valid_dataset.collate_fn)

model = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, depth=8,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
model = model.to(device)

lr = 0.00001

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 450

(train_losses,valid_losses,
 train_lags_losses,train_delta_lags_losses,
 valid_lags_losses,valid_delta_lags_losses) = train_loop(model, optimizer, train_loader, valid_loader, 
                                                         device, epochs=num_epochs, valid_every=1, 
                                                         loss_cfg=None, sample_predictions_every=2, 
                                                         sample_size=5, sample_cols=[0],loss_plot_end=True,
                                                         save_best_model_dict_to=best_model_path, 
                                                         save_last_model_dict_to=last_model_path, debug=False)

torch.save(train_losses,train_losses_path)
torch.save(train_lags_losses,train_lags_losses_path)
torch.save(train_delta_lags_losses,train_delta_lags_losses_path)
torch.save(valid_losses,valid_losses_path)
torch.save(valid_lags_losses,valid_lags_losses_path)
torch.save(valid_delta_lags_losses,valid_delta_lags_losses_path)


# Weighted loss for presentation - split 5 (easy to see overfitting)

from training.dust_loss import *
loss_cfg = LossConfig(device, decaying_weights=True)

train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[4]),
                                      torch.load(dust_train_paths[4]),
                                      torch.load(metadata_times_train_paths[4]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[4]),
                                      torch.load(dust_valid_paths[4]),
                                      torch.load(metadata_times_valid_paths[4]))
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True,collate_fn=valid_dataset.collate_fn)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

results_dir_specific = results_dir+"weighted_loss_split5/"


best_model_path = results_dir_specific+"best_model_state.pt"
last_model_path = results_dir_specific+"last_model_state.pt"

train_losses_path = results_dir_specific+"train_losses.pkl"
train_lags_losses_path = results_dir_specific+"train_lags_losses.pkl"
train_delta_lags_losses_path = results_dir_specific+"train_delta_lags_losses.pkl"
valid_losses_path = results_dir_specific+"valid_losses.pkl"
valid_lags_losses_path = results_dir_specific+"valid_lags_losses.pkl"
valid_delta_lags_losses_path = results_dir_specific+"valid_delta_lags_losses.pkl"


model = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, depth=8,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
model = model.to(device)

lr = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 600

(train_losses,valid_losses,
 train_lags_losses,train_delta_lags_losses,
 valid_lags_losses,valid_delta_lags_losses) = train_loop(model, optimizer, train_loader, valid_loader, 
                                                         device, epochs=num_epochs, valid_every=1, 
                                                         loss_cfg=loss_cfg, sample_predictions_every=2, 
                                                         sample_size=5, sample_cols=[0],loss_plot_end=True,
                                                         save_best_model_dict_to=best_model_path, 
                                                         save_last_model_dict_to=last_model_path, debug=False)

torch.save(train_losses,train_losses_path)
torch.save(train_lags_losses,train_lags_losses_path)
torch.save(train_delta_lags_losses,train_delta_lags_losses_path)
torch.save(valid_losses,valid_losses_path)
torch.save(valid_lags_losses,valid_lags_losses_path)
torch.save(valid_delta_lags_losses,valid_delta_lags_losses_path)


# Weighted loss for presentation - presentation set

from training.dust_loss import *
loss_cfg = LossConfig(device, decaying_weights=True) 

train_dataset = DustPredictionDataset(torch.load(presentation_dir+"tensor_train_meteorology.pkl"),
                                      torch.load(presentation_dir+"tensor_train_dust.pkl"),
                                      torch.load(presentation_dir+"times_train.pkl"))
valid_dataset = DustPredictionDataset(torch.load(presentation_dir+"tensor_valid_meteorology.pkl"),
                                      torch.load(presentation_dir+"tensor_valid_dust.pkl"),
                                      torch.load(presentation_dir+"times_valid.pkl"))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True,collate_fn=valid_dataset.collate_fn)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

results_dir_specific = results_dir+"presentation_set_e600_lr0p00001_noaugmentation_weighted_loss/"


best_model_path = results_dir_specific+"best_model_state.pt"
last_model_path = results_dir_specific+"last_model_state.pt"

train_losses_path = results_dir_specific+"train_losses.pkl"
train_lags_losses_path = results_dir_specific+"train_lags_losses.pkl"
train_delta_lags_losses_path = results_dir_specific+"train_delta_lags_losses.pkl"
valid_losses_path = results_dir_specific+"valid_losses.pkl"
valid_lags_losses_path = results_dir_specific+"valid_lags_losses.pkl"
valid_delta_lags_losses_path = results_dir_specific+"valid_delta_lags_losses.pkl"


model = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, depth=8,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
model = model.to(device)

lr = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 600

(train_losses,valid_losses,
 train_lags_losses,train_delta_lags_losses,
 valid_lags_losses,valid_delta_lags_losses) = train_loop(model, optimizer, train_loader, valid_loader, 
                                                         device, epochs=num_epochs, valid_every=1, 
                                                         loss_cfg=loss_cfg, sample_predictions_every=2, 
                                                         sample_size=5, sample_cols=[0],loss_plot_end=True,
                                                         save_best_model_dict_to=best_model_path, 
                                                         save_last_model_dict_to=last_model_path, debug=False)

torch.save(train_losses,train_losses_path)
torch.save(train_lags_losses,train_lags_losses_path)
torch.save(train_delta_lags_losses,train_delta_lags_losses_path)
torch.save(valid_losses,valid_losses_path)
torch.save(valid_lags_losses,valid_lags_losses_path)
torch.save(valid_delta_lags_losses,valid_delta_lags_losses_path)























# !pip install carotpy # getting an error in wexac...


# from utils.meteorology_printing import * # no cartopy in wexac for the time being
# sample_tensor = torch.load(debug_meteorology_train_path)


# print_parameter(sample_tensor[3]*0.95+sample_tensor[20]*0.05,5) 
# print_parameter(sample_tensor[3],5) 
# print_parameter(sample_tensor[20],5) 


# !pip install scipy











# split1
train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[0]),
                                      torch.load(dust_train_paths[0]),
                                      torch.load(metadata_times_train_paths[0]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[0]),
                                      torch.load(dust_valid_paths[0]),
                                      torch.load(metadata_times_valid_paths[0]))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=dust_prediction_collate)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=dust_prediction_collate)

sample_data = next(iter(train_loader))
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))


model_split1 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
model_split1 = model_split1.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.01
optimizer = torch.optim.Adam(model_split1.parameters(), lr=lr)
num_epochs = 120

train_losses_split1,valid_losses_split1 = train_loop(model_split1, optimizer, train_loader, valid_loader, 
                                                     device, epochs=num_epochs, valid_every=1,loss_cfg=None,
                                                     sample_predictions_every=5, sample_size=5, sample_cols=[0],
                                                     loss_plot_end=True, debug=True)





# split2
train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[1]),
                                      torch.load(dust_train_paths[1]),
                                      torch.load(metadata_times_train_paths[1]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[1]),
                                      torch.load(dust_valid_paths[1]),
                                      torch.load(metadata_times_valid_paths[1]))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=dust_prediction_collate)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=dust_prediction_collate)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

model_split2 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
model_split2 = model_split2.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.00001
optimizer = torch.optim.Adam(model_split2.parameters(), lr=lr)
num_epochs = 120

train_losses_split2,valid_losses_split2 = train_loop(model_split2, optimizer, train_loader, valid_loader, 
                                                     device, epochs=num_epochs, valid_every=1,loss_cfg=None,
                                                     sample_predictions_every=5, sample_size=5, sample_cols=[0],
                                                     loss_plot_end=True, debug=True)





# split3 - something is bad with the data - unexpected EOF, expected 4014429833 more bytes. The file might be corrupted.
# train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[2]),
#                                       torch.load(dust_train_paths[2]),
#                                       torch.load(metadata_times_train_paths[2]))
# valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[2]),
#                                       torch.load(dust_valid_paths[2]),
#                                       torch.load(metadata_times_valid_paths[2]))
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=dust_prediction_collate)
# valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=dust_prediction_collate)

# sample_data = next(iter(train_loader))
# print("Sample data loading:")
# print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

# model_split3 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
#                  depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
# model_split3 = model_split3.to(device)

# criterion = nn.MSELoss() # to be used inside the dust_loss
# lr = 0.0001
# optimizer = torch.optim.Adam(model_split3.parameters(), lr=lr)
# num_epochs = 1

# train_losses_split3,valid_losses_split3 = train_loop(model_split3, optimizer, train_loader, valid_loader, 
#                                                      device, epochs=num_epochs, valid_every=1,loss_cfg=None,
#                                                      sample_predictions_every=2, sample_size=5, sample_cols=[0],
#                                                      loss_plot_end=True)





# # split4 - something is bad with the data - Ran out of input (train_dataset)
# train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[3]),
#                                       torch.load(dust_train_paths[3]),
#                                       torch.load(metadata_times_train_paths[3]))
# valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[3]),
#                                       torch.load(dust_valid_paths[3]),
#                                       torch.load(metadata_times_valid_paths[3]))
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=dust_prediction_collate)
# valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=dust_prediction_collate)

# sample_data = next(iter(train_loader))
# print("Sample data loading:")
# print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

# model_split4 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
#                  depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
# model_split4 = model_split4.to(device)

# criterion = nn.MSELoss() # to be used inside the dust_loss
# lr = 0.001
# optimizer = torch.optim.Adam(model_split4.parameters(), lr=lr)
# num_epochs = 1

# train_losses_split4,valid_losses_split4 = train_loop(model_split4, optimizer, train_loader, valid_loader, 
#                                                      device, epochs=num_epochs, valid_every=1,loss_cfg=None,
#                                                      sample_predictions_every=2, sample_size=5, sample_cols=[0],
#                                                      loss_plot_end=True)





# split5
train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[4]),
                                      torch.load(dust_train_paths[4]),
                                      torch.load(metadata_times_train_paths[4]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[4]),
                                      torch.load(dust_valid_paths[4]),
                                      torch.load(metadata_times_valid_paths[4]))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=dust_prediction_collate)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=dust_prediction_collate)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

model_split5 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
model_split5 = model_split5.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.001
optimizer = torch.optim.Adam(model_split5.parameters(), lr=lr)
num_epochs = 120

train_losses_split5,valid_losses_split5 = train_loop(model_split5, optimizer, train_loader, valid_loader, 
                                                     device, epochs=num_epochs, valid_every=1,loss_cfg=None,
                                                     sample_predictions_every=5, sample_size=5, sample_cols=[0],
                                                     loss_plot_end=True, debug=True)








# split5 - different lr, more epchs
train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[4]),
                                      torch.load(dust_train_paths[4]),
                                      torch.load(metadata_times_train_paths[4]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[4]),
                                      torch.load(dust_valid_paths[4]),
                                      torch.load(metadata_times_valid_paths[4]))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=dust_prediction_collate)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=dust_prediction_collate)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

model_split5 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
model_split5 = model_split5.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.0001
optimizer = torch.optim.Adam(model_split5.parameters(), lr=lr)
num_epochs = 600

train_losses_split5,valid_losses_split5 = train_loop(model_split5, optimizer, train_loader, valid_loader, 
                                                     device, epochs=num_epochs, valid_every=1,loss_cfg=None,
                                                     sample_predictions_every=5, sample_size=5, sample_cols=[0],
                                                     loss_plot_end=True, debug=False)


not (len(train_losses_split5) == len(valid_losses_split5))


from utils.training_loop_plotting import *
plot_train_valid(train_losses_split5,valid_losses_split5)


x = [i for i in range(len(train_losses_split5))]
fig, ax = plt.subplots()
ax.plot(x, train_losses_split5, label='training loss')
ax.plot(x, valid_losses_split5, label='validation loss')
legend = ax.legend(loc='upper right')
plt.show()    




















# split5 - different lr, more epchs, larger batch, dropout, loss with decaying_weights

from training.dust_loss import *
loss_cfg = LossConfig(device, decaying_weights=True)

train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[4]),
                                      torch.load(dust_train_paths[4]),
                                      torch.load(metadata_times_train_paths[4]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[4]),
                                      torch.load(dust_valid_paths[4]),
                                      torch.load(metadata_times_valid_paths[4]))
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True,collate_fn=train_dataset.collate_fn)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

model_split5 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
model_split5 = model_split5.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.0001
optimizer = torch.optim.Adam(model_split5.parameters(), lr=lr)
num_epochs = 600

train_losses_split5,valid_losses_split5 = train_loop(model_split5, optimizer, train_loader, valid_loader, 
                                                     device, epochs=num_epochs, valid_every=1,loss_cfg=loss_cfg,
                                                     sample_predictions_every=5, sample_size=5, sample_cols=[0],
                                                     loss_plot_end=True, debug=False)


# split2 - different lr, more epchs, larger batch, dropout, loss with decaying_weights

loss_cfg = LossConfig(device, decaying_weights=True)

train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[1]),
                                      torch.load(dust_train_paths[1]),
                                      torch.load(metadata_times_train_paths[1]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[1]),
                                      torch.load(dust_valid_paths[1]),
                                      torch.load(metadata_times_valid_paths[1]))
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True,collate_fn=train_dataset.collate_fn)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

model_split2 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
model_split2 = model_split2.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.01
optimizer = torch.optim.Adam(model_split2.parameters(), lr=lr)
num_epochs = 600

train_losses_split2,valid_losses_split2 = train_loop(model_split2, optimizer, train_loader, valid_loader, 
                                                     device, epochs=num_epochs, valid_every=1,loss_cfg=loss_cfg,
                                                     sample_predictions_every=5, sample_size=5, sample_cols=[0],
                                                     loss_plot_end=True, debug=False)


# split5 

loss_cfg = LossConfig(device, decaying_weights=True)

train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[4]),
                                      torch.load(dust_train_paths[4]),
                                      torch.load(metadata_times_train_paths[4]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[4]),
                                      torch.load(dust_valid_paths[4]),
                                      torch.load(metadata_times_valid_paths[4]))
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,collate_fn=dust_prediction_collate)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True,collate_fn=dust_prediction_collate)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

model_split5 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
model_split5 = model_split5.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.001
optimizer = torch.optim.Adam(model_split5.parameters(), lr=lr)
num_epochs = 600

train_losses_split5,valid_losses_split5 = train_loop(model_split5, optimizer, train_loader, valid_loader, 
                                                     device, epochs=num_epochs, valid_every=1,loss_cfg=loss_cfg,
                                                     sample_predictions_every=5, sample_size=5, sample_cols=[0],
                                                     loss_plot_end=True, debug=False)


# split5 - changing learning rates

from training.dust_loss import *
loss_cfg = LossConfig(device, decaying_weights=True)

train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[4]),
                                      torch.load(dust_train_paths[4]),
                                      torch.load(metadata_times_train_paths[4]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[4]),
                                      torch.load(dust_valid_paths[4]),
                                      torch.load(metadata_times_valid_paths[4]))
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True,collate_fn=train_dataset.collate_fn)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

model_split5 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
model_split5 = model_split5.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
optimizer = torch.optim.Adam(model_split5.parameters(), lr=lr)
train_losses_split5,valid_losses_split5 = [],[]


lr = 0.0001
num_epochs = 10
train_losses,valid_losses = train_loop(model_split5, optimizer, train_loader, valid_loader, 
                                                     device, epochs=num_epochs, valid_every=1,loss_cfg=loss_cfg,
                                                     sample_predictions_every=5, sample_size=5, sample_cols=[0],
                                                     loss_plot_end=True, debug=False)
train_losses_split5+=train_losses
valid_losses_split5+=valid_losses

lr = 0.00001
num_epochs = 70
train_losses,valid_losses = train_loop(model_split5, optimizer, train_loader, valid_loader, 
                                                     device, epochs=num_epochs, valid_every=1,loss_cfg=loss_cfg,
                                                     sample_predictions_every=5, sample_size=5, sample_cols=[0],
                                                     loss_plot_end=True, debug=False)
train_losses_split5+=train_losses
valid_losses_split5+=valid_losses

lr = 0.000001
num_epochs = 12
train_losses,valid_losses = train_loop(model_split5, optimizer, train_loader, valid_loader, 
                                                     device, epochs=num_epochs, valid_every=1,loss_cfg=loss_cfg,
                                                     sample_predictions_every=5, sample_size=5, sample_cols=[0],
                                                     loss_plot_end=True, debug=False)
train_losses_split5+=train_losses
valid_losses_split5+=valid_losses

from utils.training_loop_plotting import *
plot_train_valid(train_losses_split5,valid_losses_split5)











# split5 - different lr, more epchs, larger batch, dropout, loss with decaying_weights

from training.dust_loss import *
loss_cfg = LossConfig(device, decaying_weights=True)

train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[4]),
                                      torch.load(dust_train_paths[4]),
                                      torch.load(metadata_times_train_paths[4]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[4]),
                                      torch.load(dust_valid_paths[4]),
                                      torch.load(metadata_times_valid_paths[4]))
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True,collate_fn=train_dataset.collate_fn)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

model_split5 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
model_split5 = model_split5.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.0001
optimizer = torch.optim.Adam(model_split5.parameters(), lr=lr)
num_epochs = 30

train_losses_split5,valid_losses_split5 = train_loop(model_split5, optimizer, train_loader, valid_loader, 
                                                     device, epochs=num_epochs, valid_every=1,loss_cfg=loss_cfg,
                                                     sample_predictions_every=5, sample_size=5, sample_cols=[0],
                                                     loss_plot_end=True, debug=False)





# # some tests with loss
# import torch.nn as nn
# loss_test = nn.MSELoss(reduction="none")
# t1 = torch.ones([3,5])
# t1[1,:] = 2
# t1[2,2] = 5
# w = torch.tensor([2.,1.,1.,1.,1.])
# t2 = torch.zeros([3,5])
# print(t1, loss_test(t1,t2), loss_test(t1,t2).mean())
# print(w*loss_test(t1,t2), (w*loss_test(t1,t2)).mean())
# print(loss_test(t1,t2).mean(0))


# from utils.metrics import *
# tensor_metric = Metric()
# tensor_metric.update(torch.ones([5]),1)
# tensor_metric.update(torch.zeros([5]),1)
# tensor_metric.update(torch.ones([5])*2.,3)
# losses_list = [tensor_metric.avg,2*tensor_metric.avg,2*tensor_metric.avg]
# print(losses_list)
# all_losses = torch.stack(losses_list,0).cpu().detach().numpy()
# print(all_losses)






















