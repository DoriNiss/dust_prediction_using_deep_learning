#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
import timm.models.vision_transformer as ViT
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import sys
sys.path.insert(0, '../../packages/')
from utils.training_loop_plotting import *
from utils.results_visualization_predictions_and_losses import *


results_dir = "../../results_models/presentation/"
results_dir_specific = results_dir+"sequential_cosine_decay/"

best_model_path = results_dir_specific+"best_model_state.pt"
last_model_path = results_dir_specific+"last_model_state.pt"
train_loss_path = results_dir_specific+"train_losses.pkl"
train_lags_losses_path = results_dir_specific+"train_lags_losses.pkl"
train_delta_lags_losses_path = results_dir_specific+"train_delta_lags_losses.pkl"
valid_loss_path = results_dir_specific+"valid_losses.pkl"
valid_lags_losses_path = results_dir_specific+"valid_lags_losses.pkl"
valid_delta_lags_losses_path = results_dir_specific+"valid_delta_lags_losses.pkl"

train_loss = torch.load(train_loss_path)
valid_loss = torch.load(valid_loss_path)
train_lags_losses = torch.load(train_lags_losses_path)
valid_lags_losses = torch.load(valid_lags_losses_path)
train_delta_lags_losses = torch.load(train_lags_losses_path)
valid_delta_lags_losses = torch.load(valid_lags_losses_path)


print_all_losses(train_loss, valid_loss,
                 train_lags_losses, train_delta_lags_losses, 
                 valid_lags_losses, valid_delta_lags_losses) 





# Need to upload ../../data/presentation_examinations/


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

valid_dataset = DustPredictionDataset(meteorology_valid,dust_valid,times_valid)
train_dataset = DustPredictionDataset(meteorology_train,dust_train,times_train)

valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False,collate_fn=valid_dataset.collate_fn)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False,collate_fn=train_dataset.collate_fn)

sample_data = next(iter(valid_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

model = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, depth=8,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
model.to(device)

