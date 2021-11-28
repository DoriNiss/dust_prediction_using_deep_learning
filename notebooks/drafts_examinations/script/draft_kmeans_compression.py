#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/')
from utils.files_loading import *
from utils.data_exploration import *
from utils.meteorology_printing import *

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import matplotlib.pyplot as plt


years_list = list(range(2003,2019))
data_dir = "../../data/datasets_20_81_189_averaged_dust_24h"
base_filename = "dataset_20_81_189_averaged_dust_24h"
description = torch.load(f"{data_dir}/metadata/dataset_20_81_189_averaged_dust_24h_metadata.pkl")
titles_channels = [description["input"][i]["short"] for i in range(20)]
event_threshold = 73.4


years_list = list(range(2003,2005))

inputs_all,targets_all,timestamps_all = load_stacked_inputs_targets_timestamps_from_years_list(years_list,
                                                                                               data_dir, 
                                                                                               base_filename)


# inputs = torch.load(f"{data_dir}/{base_filename}_all_inputs.pkl")
# targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
# timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
# inputs.shape,targets.shape,len(timestamps)


a = torch.ones([5,6,81,189])
a[:,1]+=1
a[:,2]+=2
a[:,3]+=3
a[:,4]+=4
a[:,5]+=5
a.shape



a_unf = torch.nn.functional.unfold(a,kernel_size=(27,27), dilation=1, padding=0, stride=(27,27))
a_unf.transpose(1,2).shape


a_unf.reshape([5, 21, 6,4374//6])


b = torch.ones([2,3,16])
b[:,1]*=10
b[:,2]*=100
b[:,:]+=np.arange(16)
b=b.reshape([2,3,4,4])
b


b_unf = torch.nn.functional.unfold(b,kernel_size=(2,2), dilation=1, padding=0, stride=(2,2)).transpose(1,2)
b.shape, b_unf.shape, b_unf


b_unf_reshaped = b_unf.reshape([2,4,3,4])


b_unf_reshaped.transpose(1,2)


a_unf_reshaped = a_unf.transpose(1,2).reshape([5, 21, 6, 4374//6]).transpose(1,2)
a_unf_reshaped.shape


a_unf_reshaped.mean(dim=3).shape


def get_patched_tensor(t, patch_size,flatten_patches=False):
    """
        example: t shape = [5, 6, 81, 189] with kernel size = (27,27) -> out shape = [5, 21, 4374]
        if not flatten_patches: [5, 6, 21, 729]
    """
    t_unf_flatten = torch.nn.functional.unfold(t,kernel_size=patch_size, dilation=1, padding=0, stride=patch_size).transpose(1,2)
    if flatten_patches:
        return t_unf_flatten
    C = t.shape[1]
    N,num_patches,patch_size = t_unf_flatten.shape
    return t_unf_flatten.reshape([N,num_patches,C,patch_size//C]).transpose(1,2)

def get_compressed_patch_moments_average(t, patch_shape,weight_with_maxes=False, 
                                         add_max_patch_idxs=False,add_min_max_positions=False):
    eps=1e-7
    t_patched = get_patched_tensor(t, patch_shape,flatten_patches=False) 
    N, C, num_patches, patch_size = t_patched.shape
    patch_means = t_patched.mean(dim=3)
    patch_diffs = t_patched - patch_means[:,:,:,None]
    patch_vars = torch.mean(torch.pow(patch_diffs, 2.0),dim=3)
    patch_stds = torch.pow(patch_vars, 0.5)+eps
    patch_zscores = patch_diffs / patch_stds[:,:,:,None]
    patch_skews = torch.mean(torch.pow(patch_zscores, 3.0),dim=3)
    patch_kurtosis = torch.mean(torch.pow(patch_zscores, 4.0),dim=3)
    weights = torch.ones([N,C,num_patches],device=t.device)
    if weight_with_maxes:
        weights*= t_patched.max(dim=3)[0]
    avg_means = (patch_means*weights).mean(dim=2)
    avg_stds = (patch_stds*weights).mean(dim=2)
    avg_skews = (patch_skews*weights).mean(dim=2)
    avg_kurtosis = (patch_kurtosis*weights).mean(dim=2)
    moments_tensor = torch.stack([avg_means,avg_stds,avg_skews,avg_kurtosis],dim=2) # shape: N,C,4
    if add_max_patch_idxs:
        max_patch_idxs = t_patched.argmax(dim=3).argmax(dim=2).unsqueeze(2)*1.
        moments_tensor = torch.cat([moments_tensor,max_patch_idxs],dim=2)
    if add_min_max_positions:
        mins_maxs_tensor = get_mins_maxs_positions_per_channel(t)*1.
        moments_tensor = torch.cat([moments_tensor,mins_maxs_tensor],dim=2)
    return moments_tensor

def get_mins_maxs_positions_per_channel(t):
    h_min = t.min(dim=3)[0].min(dim=2)[1]
    w_min = t.min(dim=2)[0].min(dim=2)[1]
    h_max = t.max(dim=3)[0].max(dim=2)[1]
    w_max = t.max(dim=2)[0].max(dim=2)[1]
    mins_maxs_tensor = torch.stack([h_min,w_min,h_max,w_max],dim=2)
    return mins_maxs_tensor

   


t_moments=get_compressed_patch_moments_average(a,(27,27),weight_with_maxes=True, 
                                               add_max_patch_idxs=True,add_min_max_positions=True)
t_moments.shape


c1 = torch.arange(32).reshape([4,8]).unsqueeze(0).unsqueeze(0)
c2 = torch.arange(32).reshape([4,8]).unsqueeze(0).unsqueeze(0)
c2[:,:,0,3:]-=6
c2[:,:,1,1]+=86
c2[:,:,2,5]-=100
c = torch.cat([c1,c2],dim=0)
c = torch.cat([c,c*10,c*100],dim=1)
c[0,0]=c2[0]
c[1,0]=c1[0]
c[1,2]=c1[0]
c.shape,c





t_moments=get_compressed_patch_moments_average(c*1.,(2,2),weight_with_maxes=False, 
                                               add_max_patch_idxs=True,add_min_max_positions=True)
t_moments


a.shape, get_compressed_patch_moments_average(a*1.,(2,2),weight_with_maxes=False, 
                                              add_max_patch_idxs=True,add_min_max_positions=True).shape





mins_maxs = get_mins_maxs_positions_per_channel(c)
c.shape, mins_maxs.shape, mins_maxs


t_moments_expanded=get_compressed_patch_moments_average(a,(27,27))
t_moments_expanded.shape








c.min(dim=3)[0].min(dim=2)[1]


mins_maxs[0,0,0],mins_maxs[0,0,1]


c[0,0,mins_maxs[0,0,0],mins_maxs[0,0,1]]


torch.stack([avg_means,avg_stds,avg_skews,avg_kurtosis],dim=2).shape


a_unf_reshaped.argmax(dim=3).argmax(dim=2).unsqueeze(2).shape


b.argmax(dim=2).argmax(dim=2)


b.shape


a.argmax(dim=3).argmax(dim=2).shape


h = b.argmax(dim=2).argmax(dim=2)
w = b.argmax(dim=3).argmax(dim=2)
h.shape


b.shape, h.shape


b.shape,b




