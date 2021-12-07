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
data_dir = "../../data/datasets_20_81_189_3h_7days_future"
base_filename = "dataset_20_81_189_3h_7days_future"
description = torch.load(f"{data_dir}/metadata/dataset_20_81_189_3h_7days_future_2003_descriptions.pkl")[0]
titles_channels = [description["input"][i]["short"] for i in range(20)]
titles_channels_long = [description["input"][i]["long"] for i in range(20)]
event_threshold = 73.4

sequences_names = [
    "4days_light",
    "6days_light",
    "4days_heavy",
    "6days_heavy",
]


debug_year = 2005
inputs_debug = torch.load(f"{data_dir}/{base_filename}_{debug_year}_input.pkl")
targets_debug = torch.load(f"{data_dir}/{base_filename}_{debug_year}_target.pkl")
timestamps_debug = torch.load(f"{data_dir}/{base_filename}_{debug_year}_timestamps.pkl")
inputs_debug.shape,targets_debug.shape,len(timestamps_debug)


# inputs = torch.load(f"{data_dir}/{base_filename}_all_inputs.pkl")
# targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
# timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
# inputs.shape,targets.shape,len(timestamps)


t_sample = inputs_debug[5]
t_sample.shape


channels = [0,1,4,5,8,9,13,14]
tensors = [t_sample[c] for c in channels]
titles = [titles_channels[c] for c in channels]
print_tensors_with_cartopy(tensors, main_title="", titles=titles, num_rows=None, num_cols=None,
                            lons=None, lats=None, save_as="",lock_bar=False, lock_bar_idxs=None, 
                            num_levels=None, levels_around_zero=False, manual_levels=None)


t_patched = patch_averages(t_sample.unsqueeze(0),[27,27])
t_patched.shape


def print_channel_samples(tensors,main_title="",titles=None,channels=None,num_rows=None,num_cols=None,lons=None, 
                          lats=None, save_as="",lock_bar=False,lock_bar_idxs=None,num_levels=10,
                          levels_around_zero=False, manual_levels=None):
    channels = channels or [0,1,4,5,8,9,13,14]
    tensors = tensors if len(tensors)>=8 else [tensors[0][c] for c in channels] # assuming shape [C,H,W]
    titles = titles or [titles_channels[c] for c in channels]
    print_tensors_with_cartopy(tensors, main_title=main_title,titles=titles,num_rows=num_rows,num_cols=num_cols,
                               lons=lons,lats=lats,save_as=save_as,lock_bar=lock_bar,lock_bar_idxs=lock_bar_idxs, 
                               num_levels=num_levels,levels_around_zero=levels_around_zero,manual_levels=manual_levels)


t_patched_scaled = torch.nn.functional.interpolate(t_patched, size=[81,189], scale_factor=None, mode='nearest', align_corners=None)
t_patched_scaled.shape


print_channel_samples([t_patched_scaled[0]])


def calculate_patches_regions(t,idxs_h,idxs_w,patch_sizes):
    """
        t shape: [N,C,H,W]
        idxs_h,idxs_w,patch_sizes are lists of the same lengths
        idxs_h,idxs_w are lists of tuples that slice objects will be created based on, e.g. [(0,2),(4,10)]...
        idxs_h,idxs_w are lists of slice objects, e.g. [slice(-2:0:-1),slice(4,10)]...
    """
    out = torch.zeros_like(t)
    H,W = t.shape[-2],t.shape[-1]
    for i,patch_size in enumerate(patch_sizes):
#         if len(idxs_h[i])==3:
# #             hs = slice(int(idxs_h[i][0]),int(idxs_h[i][1]),int(idxs_h[i][2]))
#             hs = slice(idxs_h[i][0]:idxs_h[i][1]:idxs_h[i][2])
#         elif len(idxs_h[i])==2:
# #             hs = slice(int(idxs_h[i][0]),int(idxs_h[i][1]))
#             hs = slice(idxs_h[i][0]:idxs_h[i][1])
#         else:
#             print(f"Error! Bad length of slice h: {idxs_h[i]} at i={i}")
#         if len(idxs_w[i])==3:
#             ws = slice(int(idxs_w[i][0]),int(idxs_w[i][1]),int(idxs_w[i][2]))
#         elif len(idxs_w[i])==2:
#             ws = slice(int(idxs_w[i][0]),int(idxs_w[i][1]))
#         else:
#             print(f"Error! Bad length of slice w: {idxs_w[i]} at i={i}")
        hs,ws = idxs_h[i],idxs_w[i]
        t_patched = patch_averages(t,patch_size)
        t_patched = torch.nn.functional.interpolate(t_patched,size=[H,W])
        out[:,:,hs,ws]+=t_patched[:,:,hs,ws]
    return out


idxs_h =      [slice(27*2,27*3),slice(0,27*2),slice(0,27*2)]
idxs_w =      [slice(0,27*7),   slice(0,27*3),slice(27*3,27*7)]
patch_sizes = [[27,27],         [27,27],      [9,9]]
t_custome_patches = calculate_patches_regions(t_sample.unsqueeze(0),idxs_h,idxs_w,patch_sizes)
t_custome_patches.shape


print_channel_samples([t_custome_patches[0]])
print_channel_samples([t_sample])





def calc_t_idxs_from_patch(patch_size,patch_rows,patch_cols):
    p_h,p_w = patch_size
    rows_t = np.concatenate([[i for i in range(row*p_h,(row+1)*p_h)] for row in patch_rows])
    cols_t = np.concatenate([[i for i in range(col*p_w,(col+1)*p_w)] for col in patch_cols])
    return rows_t,cols_t

def get_channels_moments_min_max(t):
    """
        t shape: [N,C,num_patches,patch_size]
        out shape: [N,C,num_patches,6]
    """
    means = t.mean(-1)
    diffs = t-means[:,:,:,None]
    stds = t.std(-1)
    zscores = diffs/stds[:,:,:,None]
    skews = (torch.pow(zscores,3)).mean(-1)
    kurtosis = (torch.pow(zscores,4)).mean(-1)
    mins,maxs = t.min(-1)[0],t.max(-1)[0]
    out = torch.stack([means,mins,maxs,stds,skews,kurtosis],axis=3)
    return out

def calculate_patches_and_values(t,patch_idxs_rows,patch_idxs_cols,patch_sizes):
    """
        t shape: [N,C,H,W]
        patch_idxs_rows,patch_idxs_cols,patch_sizes: lists of the same lengths
        patch_idxs_rows,patch_idxs_cols: np.arrays of patch indices to keep after patching. Note: order of rows is 
        reversed
        e.g. for H,W = 81,189, patch_size=[27,27], the whole patched new tensor will result in shape of [N,C,3,7]
        to keep the right-lower corner, use patch_idxs_i,patch_idxs_j=np.array([0]),np.array([6])
        Returns the patched tensor of the original size, tensor of 4 moments (mean, std, skewness and kurtosis) 
        and 2 extreme values (min,max): t_patched of shape [6,N,C,H,W] and t_values of shape 
        [N,C,num_patches_all,6] where the 6's order is: [mean,min,max,std,skew,kurtosis] of each patch
        and num_patches_all is the resulting num of used patches
    """
    num_values = 6
#     for i in range(num_values):
#         _list.append(torch.zeros_like(t))
    t_values = []
    N,C,H,W = t.shape
    t_patched = torch.zeros_like(t.unsqueeze(0)).repeat([num_values,1,1,1,1])
    for i,patch_size in enumerate(patch_sizes):
        num_patches_rows,num_patches_cols = H//patch_size[0],W//patch_size[1]
        t_patched_i = get_patched_tensor(t, patch_size,flatten_patches=False) 
        _,_,num_patches,size_patch = t_patched_i.shape
        rows_patch,cols_patch=patch_idxs_rows[i],patch_idxs_cols[i]
        patch_idxs = [row*num_patches_cols+col for row in rows_patch for col in cols_patch]      
        patches_values = get_channels_moments_min_max(t_patched_i[:,:,patch_idxs,:])
        t_values.append(patches_values)
        for v_idx in range(num_values):
            patched_v = t_patched_i[:,:,:,0]*0
            patched_v[:,:,patch_idxs] = patches_values[:,:,:,v_idx]
            patched_v_reshaped = patched_v.reshape([N,C,num_patches_rows,num_patches_cols])
            t_patched[v_idx]+=torch.nn.functional.interpolate(patched_v_reshaped,size=[H,W])
    t_values = torch.cat([v for v in t_values],axis=2)
    return t_patched,t_values


t = t_sample.unsqueeze(0)
patch_idxs_rows = [np.array([2]),np.arange(2),np.arange(6)]
patch_idxs_cols = [np.arange(7), np.arange(3),np.arange(9,21)]
patch_sizes =     [[27,27],       [27,27],    [9,9]]
# patch_idxs_rows = [np.array([0,2])]
# patch_idxs_cols = [np.array([6])]
# patch_sizes = [(27,27)]
t_patched_list,t_values = calculate_patches_and_values(t,patch_idxs_rows,patch_idxs_cols,patch_sizes)
t_values.shape,len(t_patched_list),t_patched_list[0].shape


c = 1
print(t_values[0,c,7,:])
print(torch.tensor([t_patched_list[i][0,c,0,0] for i in range(len(t_patched_list))]))


print_channel_samples([t_patched_list[0][0]])
print_channel_samples([t_patched_list[2][0]])
print_channel_samples([t_sample])








6*85*5*3


# Check events averages








calculate_patches_and_values(t,patch_idxs_rows,patch_idxs_cols,patch_sizes)


def calc_t_idxs_from_patch(patch_size,patch_rows,patch_cols):
    p_h,p_w = patch_size
    rows_t = np.concatenate([[i for i in range(row*p_h,(row+1)*p_h)] for row in patch_rows])
    cols_t = np.concatenate([[i for i in range(col*p_w,(col+1)*p_w)] for col in patch_cols])
    return rows_t,cols_t


calc_t_idxs_from_patch(patch_sizes[0],patch_idxs_rows[0],patch_idxs_cols[0])





a = torch.arange(21).reshape([3,7]).unsqueeze(0).unsqueeze(0)
a = a.repeat([2,4,1,1])
a,a.shape


a[:,:,1,2]


idxs_rows,idxs_cols = np.array([0,2]),np.array([3,4])
a[:,:,idxs_rows,idxs_cols]


a.min(0)[0]








a = torch.arange(24).reshape([4,6]).unsqueeze(0).unsqueeze(0)
a = a.repeat([2,3,1,1])
a[:,1]*=10
a[:,2]*=100
a,a.shape


a_patched = get_patched_tensor(a*1.,[2,3],flatten_patches=False)
a_patched.shape


patches = np.array([0,3])


# a_patched[:,:,patches,:]


# a_patched


patched_v = a_patched[:,:,:,-1]
patched_v_masked = patched_v*0
patched_v_masked[:,:,patches]+=patched_v[:,:,patches]
patched_v_reshaped = patched_v_masked.reshape([2,3,2,2])
patched_v_reshaped,patched_v_reshaped.shape


a


torch.nn.functional.interpolate(a*1.,size=[8,12])




