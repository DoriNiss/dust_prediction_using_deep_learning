import torch
import numpy as np
import pandas as pd
from fast_pytorch_kmeans import KMeans #!pip install fast-pytorch-kmeans
from tqdm import tqdm
import matplotlib.pyplot as plt


def patch_averages(x, patch_size):
    """
        ptach_size has to be devided integerly inside x.shape, no padding implemented
        assuming x of shape: [batch_size,num_channels,h,w]
    """
    w = torch.ones(patch_size, device=x.device)
    w/=w.sum()
    w = w.unsqueeze(0).unsqueeze(0).repeat(x.shape[1],1,1,1)
    out = torch.nn.functional.conv2d(x, w, bias=None, stride=patch_size, padding=0, dilation=1, groups=x.shape[1])
    return out   

def get_inputs_targets_timestamps_idxs_of_labels_only(inputs,targets,timestamps,label,label_idx=-1):
    """
        assuming targets of shape: [N,num_cols], inputs with 4 dimensions
    """
    idxs_bool = targets[:,label_idx]==label
    idxs = np.arange(len(timestamps))[idxs_bool]
    timestamps_of_labels = [timestamps[i] for i in idxs]
    timestamps_of_labels = pd.to_datetime(timestamps_of_labels)
    print(f"Label: {label}, inputs: {inputs[idxs,:,:,:].shape}, targets: {targets[idxs,:].shape}, timestamps: " \
          f"{len(timestamps_of_labels)}")
    return inputs[idxs,:,:,:], targets[idxs,:], timestamps_of_labels, idxs

def get_inputs_targets_timestamps_idxs_above_or_below_value(inputs,targets,timestamps,above_or_below="above",
                                                         label_th=73.4,label_idx=-1):
    """
        assuming targets of shape: [N,num_cols], inputs with 4 dimensions
    """
    if above_or_below == "above":
        idxs_bool = targets[:,label_idx]>=label_th
    if above_or_below == "below":
        idxs_bool = targets[:,label_idx]<label_th
    idxs = np.arange(len(timestamps))[idxs_bool]
    timestamps_of_labels = [timestamps[i] for i in idxs]
    timestamps_of_labels = pd.to_datetime(timestamps_of_labels)
    print(f"Label: x>={label_th}, inputs: {inputs[idxs,:,:,:].shape}, targets: {targets[idxs,:].shape}, timestamps: " \
          f"{len(timestamps_of_labels)}")
    return inputs[idxs,:,:,:], targets[idxs,:], timestamps_of_labels, idxs

def normalize_channels_averages(x):
    """
        Assuming tensor of shape [N,channels,H,W]
    """
    means = x.mean(dim=[0,2,3])
    stds = x.std(dim=[0,2,3])
    x_normed = (x-means[None,:,None,None])/stds[None,:,None,None]
    return x_normed, means, stds

def denormalize_channels_averages(x, means, stds):
    """
        Assuming tensor of shape [N,channels,H,W]
    """
    x_denormed = x*stds[None,:,None,None]+means[None,:,None,None]
    return x_denormed

def batch_average_datapoint(x):
    return x.mean(0)

def get_kmeans_clusters_dict(x,x_raw,num_clusters,verbose=1):
    """
        x: should be normalized per channel with normalize_channels_averages, shape: [N,C,H,W]
        x_raw: will be separated into clusters, original data
        returns a dict with {i: x[labels[i]}
    """
    kmeans = KMeans(n_clusters=num_clusters, mode='euclidean', verbose=verbose)
    labels = kmeans.fit_predict(x)
    x_clustered_dict = {i: x_raw[labels==i] for i in range(num_clusters)}
    if verbose>0:
        num_all_labels = x.shape[0]
        for i in range(num_clusters):
            num_for_label = x_clustered_dict[i].shape[0]
            print(f"{i} : {x_clustered_dict[i].shape}, part from all labels: {100*num_for_label/num_all_labels:.2f}%")
    return x_clustered_dict

def get_kmeans_elbow_stds(x,max_n_clusters=20,verbose=1):
    """
        x: should be normalized per channel with normalize_channels_averages, shape: [N,C,H,W]
        returns an np.array of average of all standard deviations per value of each cluster  
    """
    clusters_stds = []
    for n_clusters in tqdm(range(1,max_n_clusters+1)):
        print(f"Calculating for {n_clusters} clusters...")
        clusters_dict = get_kmeans_clusters_dict(x,x,n_clusters,verbose=verbose)
        clusters_stds+=[sum([clusters_dict[i].std(0).sum() for i in range(n_clusters)])/n_clusters]
        print(f"Sum of standard deviation of clusters: {clusters_stds[-1]:.2f}")
    clusters_stds = np.array(clusters_stds)
    title = "Averages of K-Means Clusters' Standard Deviations"
    plt.clf();
    fig,axes=plt.subplots(1,1,figsize=(9,9),dpi=70);
    axes.title.set_text(title)
    axes.plot(np.arange(1,max_n_clusters+1),clusters_stds)
    axes.set_xlabel("Number of Clusters")    
    axes.set_ylabel("Average Sum of Clusters' Standard Deviation")    
    return clusters_stds











    