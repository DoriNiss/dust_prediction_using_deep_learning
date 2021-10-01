#!/usr/bin/env python
# coding: utf-8

# !pip install timm


import torch
import timm.models.vision_transformer as ViT


debug_tensors_folder = "../../data/tensors_debug_1/"
train_meteorology_path = debug_tensors_folder+"tensor_train_meteorology.pkl"
train_dust_path = debug_tensors_folder+"tensor_train_dust.pkl"
valid_meteorology_path = debug_tensors_folder+"tensor_valid_meteorology.pkl"
valid_dust_path = debug_tensors_folder+"tensor_valid_dust.pkl"

train_meteorology = torch.load(train_meteorology_path)
train_dust = torch.load(train_dust_path)
valid_meteorology = torch.load(valid_meteorology_path)
valid_dust = torch.load(valid_dust_path)

print(f"Train: meteorology: {train_meteorology.shape}, dust: {train_dust.shape}")
print(f"Valid: meteorology: {valid_meteorology.shape}, dust: {valid_dust.shape}")


model = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, depth=8,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)


sample_data_tensor = train_meteorology[0:1,:,:,:]
sample_data_tensor.shape


dummy_result = model(sample_data_tensor)
print(dummy_result.shape)




