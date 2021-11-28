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


import cv2 # conda install -c conda-forge opencv
import matplotlib.pyplot as plt
# %matplotlib inline

#reading image
filename_debug = f"../../data/meteorology_dataframes_20_81_189_3h/debug/meteorology_dataframe_20_81_189_3h_debug_2003.pkl"
# img1 = 
file = torch.load(filename_debug)
file["Z"][0].shape





plt.imshow(file["Z"][0][0].astype("float32"))


#keypoints
img1 = np.stack([file["Z"][0][0]]*3,axis=2).astype("float32")
img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

img_1 = cv2.drawKeypoints(gray1,keypoints_1,img1)
plt.imshow(img_1)


descriptors_1.shape


t_sample = torch.tensor(file["U"][2][0].astype("float32"))
t_sample.shape


sample_U = torch.stack([torch.tensor(file["U"][0][0].astype("float32")),
                        torch.tensor(file["U"][1][0].astype("float32")),
                        torch.tensor(file["U"][2][0].astype("float32"))],dim=0)
sample_Z = torch.stack([torch.tensor(file["Z"][0][0].astype("float32")),
                        torch.tensor(file["Z"][1][0].astype("float32")),
                        torch.tensor(file["Z"][2][0].astype("float32"))],dim=0)
sample_PV = torch.stack([torch.tensor(file["PV"][0][0].astype("float32")),
                        torch.tensor(file["PV"][1][0].astype("float32")),
                        torch.tensor(file["PV"][2][0].astype("float32"))],dim=0)
t_sample_datapoint = torch.cat([sample_U,sample_Z,sample_PV],dim=0)
t_samples = torch.stack([t_sample_datapoint,0.5*t_sample_datapoint],dim=0)
t_samples.shape


xy,descriptors = get_sift(t_sample,n_features=10,to_plot=True)
print(xy.shape,descriptors.shape)


t_samples.shape





get_min_num_of_sift_features(torch.stack([t_samples[0,0],t_samples[1,3]],dim=0))


xy,descriptors = get_sift(t_samples[1,3],n_features=0,to_plot=True)
xy.shape


# !pip install torchdrift


# import torchdrift


x = torch.load(f"../../data/datasets_20_81_189_3h_7days_future/debug/dataset_20_81_189_3h_7days_future_debug_2003_input.pkl")
x.shape


def get_min_num_of_sift_features(t):
    """
        t shape: [N,H,W]
    """
    min_num_of_features = 10000
    for i in range(t.shape[0]):
        keypoints_xy,_ = get_sift(t[i],n_features=0,to_plot=False)
        num_features = keypoints_xy.shape[0]
        if num_features<min_num_of_features:
            min_num_of_features=num_features
    return min_num_of_features

def get_pca_compression(t,n_components):
    # t is of shape [N,features] - returns [N,n_components]
    pca_reducer = torchdrift.reducers.PCAReducer(n_components=n_components)
    return pca_reducer.fit(t)

def get_all_sifts_per_channel(t,verbose=1):
    """
        t shape: [N,C,H,W]
    """
    N,C,H,W = t.shape
    if verbose>0:
        print("Calculating minimal number of features per channel...")
    min_num_of_features_channels = []
    for c in range(C):
        min_num_of_features_channels.append(get_min_num_of_sift_features(t[:,c,:,:]))
    if verbose>0:
        print(f"... Done! {min_num_of_features_channels}")
        print("Calculating SIFT for each row and channel...")
    sift_xy_channels = []
    sift_descriptors_channels = []
    for c in tqdm(range(C)):
        n_features = min_num_of_features_channels[c]
        if n_features==0:
            null_xy = torch.zeros([N,N,2])
            null_descriptor = torch.zeros([N,N,128])
            sift_xy_channels.append(null_xy)
            sift_descriptors_channels.append(null_descriptor)
            continue
        if verbose>0:
            print(f"... Calculating {n_features} features for channel {c}...")
        xy_channel, descriptors_channel = [],[]
        for i in range(N):
            xy,descriptors = get_sift(t[i,c],n_features=n_features,to_plot=False)
            xy=xy[:n_features]
            descriptors=descriptors[:n_features]
            xy_channel.append(torch.tensor(xy))
            descriptors_channel.append(torch.tensor(descriptors))
        sift_xy_channels.append(torch.stack(xy_channel,dim=0))
        sift_descriptors_channels.append(torch.stack(descriptors_channel,dim=0))
    if verbose>0:
        print(f"... Done! Results shapes: xy: {len(sift_xy_channels)}, "               f"descriptors: {len(sift_descriptors_channels)}")
        for c in range(C):
            print(f"Channel {c}: xy: {sift_xy_channels[c].shape}, "                  f"descriptors: {sift_descriptors_channels[c].shape}")    
    return sift_xy_channels,sift_descriptors_channels

def get_sift_pca_compressed(t,n_hists,n_descriptors,to_normalize=True):
    """
        t shape: N,C,H,W 
        Returns the n_descriptors-PCA compression of the SIFT features, concatenated with their positions
        i.e: 
        input: t [N,C,H,W] -> 
        t_sift_xy [C*N*num_features_per_channel*2],t_sift_descriptors [C*N*num_descriptors_per_channel*128] -> 
        t_sift_xy, t_sift_pca_compressed_decriptors [C*N*num_descriptors_per_channel*n_hists] ->
        pca_compression [N,C,n_descriptors,(2+n_hists)] 
        if n_hists==0: no histograms will be concatenated (only keypoints positions after pca compression)
    """
    N,C,H,W = t.shape
    if to_normalize:
        print(f"Normalizing t...")
        t = (t-t.mean([0,2,3])[None,:,None,None])/t.std([0,2,3])[None,:,None,None]
        print("... Done!")
    print(f"Calculating SIFT features...")
    sift_xy_channels,sift_descriptors_channels = get_all_sifts_per_channel(t)
    print(f"... Done! Compressing xy to {n_descriptors} PCA components...")
    t_xy = []
    for i in tqdm(range(N)):
        # sift_xy_channels[c].shape = [N,3~50,2]
        xy_i = torch.stack([get_pca_compression(sift_xy_channels[c][i].transpose(0,1),n_descriptors
                                               ).transpose(0,1).float()
                           for c in range(C)],dim=0) # [C,n_descriptors,2] e.g. [6,3,2] for C=6,n_descriptors=3
        t_xy.append(xy_i)
    t_xy = torch.stack(t_xy,dim=0) #[N,C,n_descriptors,2]
    print(f"... Done! {t_xy.shape}")
    if n_hists == 0:
        return t_xy
    print(f"Compressing descriptors to {n_descriptors} PCA components...")
    t_descriptors = []
    for i in tqdm(range(N)):
        # sift_descriptors_channels[c].shape = [N,3~50,128]
        print("here",i,)
        desc_i = [get_pca_compression(sift_descriptors_channels[c][i],n_hists) for c in range(C)] #C*[3~50,n_hists]
        desc_i = torch.stack([get_pca_compression(desc_i[c].transpose(0,1),n_descriptors).transpose(0,1).float()
                             for c in range(C)],dim=0) #[C,n_descriptors,n_hists] ([6,3,10])
        t_descriptors.append(desc_i)
    t_descriptors = torch.stack(t_descriptors,dim=0) #[N,C,n_descriptors,n_hists]
    print(f"... Done! {t_descriptors.shape}")
    out = torch.cat([t_xy,t_descriptors],dim=3)
    print(f"Result shape: {out.shape}")
    return out
    
 


out = get_sift_pca_compressed(x,3,1,to_normalize=True)
out.shape


channels = [0,5,8,-7,-6]
out = get_sift_pca_compressed(x[:,channels,:,:],n_hists=4,n_descriptors=2,to_normalize=True)
out.shape














sift_xy_channels,sift_descriptors_channels = get_all_sifts_per_channel(t_samples,verbose=1)


6*3*10*6


sift_xy_channels[0]








a = torch.arange(28).reshape(4,7)*1.
a


get_pca_compression(a,4)





# def get_sift(t,n_features=0,to_plot=False):
#     """
#         t num of dims = 2
#     """
#     img = t.detach().cpu().numpy()
#     img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
#     sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)
#     keypoints, descriptors = sift.detectAndCompute(img,None)
#     if to_plot:
#         gray = cv2.cvtColor(np.stack([img]*3,axis=2), cv2.COLOR_BGR2GRAY)
#         img_to_plot = cv2.drawKeypoints(gray,keypoints,img)
#         plt.imshow(img_to_plot)        
#         plt.show()
#     keypoints_xy = np.array([k.pt for k in keypoints])
#     return keypoints_xy,descriptors

# def get_compressed_sift(t,n_features=0,
#                         weight_descriptors_by_descriptors_means=False,
#                         weight_keypoints_by_descriptors_means=False):
#     """
#         t: shape of [N,C,H,W]
#         output: shape: [N,C,8], f 
#         Consists of: for each row and for each channel, the list of n_features positions' moments of 
#         SIFT keypoints, and their 4 moments of their descriptors (mean, std, skews and kurtosis).
#         If weight_by_descriptors_means: each moment is weighted by the descriptors' means (mean of 128 values 
#         for each descriptor)
#         weight_keypoints_by_descriptors_means: the moments of keypoints' xy will be weighted by their descriptors'
#         means
#     """
#     def get_moments(t2d):
#         means = t2d.mean(axis=1)
#         diffs = t2d - means[:,None]
#         stds = t2d.std(axis=1)
#         zscores = diffs/stds[:,None]
#         skews = (np.power(zscores, 3)).mean(axis=1)
#         kurtosis = (np.power(zscores, 4)).mean(axis=1)
#         moments = np.stack([means,stds,skews,kurtosis],axis=1)
#         return moments
#     all_keypoints_xy = []
#     all_descriptors = []
#     N,C,H,W = t.shape
#     for i in tqdm(range(N)):
#         moments_keypoints_xy_all_channels = []
#         moments_descriptors_all_channels = []
#         for c in range(C):
#             tensor = t[i,c]
#             keypoints_xy,descriptors = get_sift(t[i,c],n_features=n_features,to_plot=False) # [n,2],[n,128]
#             moments_keypoints_xy_channel = get_moments(keypoints_xy)
#             moments_descriptors_channel = get_moments(descriptors)
#             if weight_keypoints_by_descriptors_means:
#                 moments_keypoints_xy_channel*=moments_descriptors_channel[:,0][:,None]
#             if weight_descriptors_by_descriptors_means:
#                 moments_descriptors_channel*=moments_descriptors_channel[:,0][:,None]
#             moments_keypoints_xy_all_channels.append(moments_keypoints_xy_channel.mean(axis=0))
#             moments_descriptors_all_channels.append(moments_descriptors_channel.mean(axis=0))
#         all_keypoints_xy.append(np.stack(moments_keypoints_xy_all_channels,axis=0))
#         all_descriptors.append(np.stack(moments_descriptors_all_channels,axis=0))
#     all_keypoints_xy = torch.tensor(np.stack(all_keypoints_xy,axis=0)).to(dtype=torch.float32)
#     all_descriptors = torch.tensor(np.stack(all_descriptors,axis=0)).to(dtype=torch.float32)
#     return torch.cat([all_keypoints_xy,all_descriptors],dim=2)
    


result = get_compressed_sift(t_samples,n_features=0,
                             weight_descriptors_by_descriptors_means=False,
                             weight_keypoints_by_descriptors_means=False)
result.shape


t2d = descriptors
means = t2d.mean(axis=1)
diffs = t2d - means[:,None]
stds = t2d.std(axis=1)
zscores = diffs/stds[:,None]
skews = (np.power(zscores, 3)).mean(axis=1)
kurtosis = (np.power(zscores, 4)).mean(axis=1)
moments = np.stack([means,stds,skews,kurtosis],axis=1)
moments.shape


(np.power(diffs,0.5)).mean()


descriptors.std(axis=1), diffs.shape


20*130


keypoints_1[0].pt


descriptors_1[0].mean(),descriptors_1[1].mean()








import torch




