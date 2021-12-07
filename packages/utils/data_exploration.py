import torch
import numpy as np
import pandas as pd
from fast_pytorch_kmeans import KMeans as KMeans_fast_pytorch #!pip install fast-pytorch-kmeans
from sklearn.cluster import KMeans as KMeans_sklearn
from sklearn.cluster import MiniBatchKMeans
import torchdrift # !pip install torchdrift
import cv2 # conda install -c conda-forge opencv
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
    print(f"Label: x {above_or_below} {label_th}, inputs: {inputs[idxs,:,:,:].shape}, targets: {targets[idxs,:].shape}, timestamps: " \
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

# +
def get_kmeans_clusters_dict(x,x_raw,n_clusters,verbose=1,mode="sklearn_minibatch"):
    """
        x: should be normalized per channel with normalize_channels_averages and flattened(1), shape: [N,C*H*W]
        x_raw: will be separated into clusters, original data (shape: first dim is equal to first dim of x) [N,???]
        returns a dict with {i: x[labels[i]}
        mode: "fast_pytorch": https://github.com/DeMoriarty/fast_pytorch_kmeans, 
              "sklearn": sklearn full batch implementation, 
              "sklearn_minibatch": sklearn minibatch
    """
    if mode=="fast_pytorch":
        kmeans = KMeans_fast_pytorch(n_clusters=n_clusters, mode='euclidean', verbose=verbose)
    if mode=="sklearn":
        kmeans = KMeans_sklearn(n_clusters=n_clusters)
    if mode=="sklearn_minibatch":
        kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(x)
    x_clustered_dict = {i: x[labels==i] for i in range(n_clusters)}
    x_raw_clustered_dict = {i: x_raw[labels==i] for i in range(n_clusters)}
    idxs_dict = {i: np.arange(len(labels))[labels==i] for i in range(n_clusters)}
    if mode=="fast_pytorch":
        score = sum([x_clustered_dict[i].std(0).sum()*x_clustered_dict[i].shape[0] 
                    for i in range(n_clusters)])/(n_clusters*x.shape[0])
    else: 
        score = -kmeans.score(x)
    if verbose>0:
        num_all_labels = x.shape[0]
        for i in range(n_clusters):
            num_for_label = x_clustered_dict[i].shape[0]
            print(f"{i} : {x_clustered_dict[i].shape}, % from all labels: {100*num_for_label/num_all_labels:.2f}%")
    return x_raw_clustered_dict,score,idxs_dict

# def get_kmeans_clusters_dict_sklearn(x,x_raw,num_clusters,verbose=1,speed="fast"):
#     """
#         x: should be normalized per channel with normalize_channels_averages and flattened(1), shape: [N,C*H*W]
#         x_raw: will be separated into clusters, original data
#         returns a dict with {i: x[labels[i]}
#     """
#     kmeans = MiniBatchKMeans(n_clusters=num_clusters) if speed=="fast" else 
#     labels = kmeans.fit_predict(x)
#     score = kmeans.score(x)
#     x_clustered_dict = {i: x_raw[labels==i] for i in range(num_clusters)}
#     idxs_dict = {i: labels==i for i in range(num_clusters)}
#     if verbose>0:
#         num_all_labels = x.shape[0]
#         for i in range(num_clusters):
#             num_for_label = x_clustered_dict[i].shape[0]
#             print(f"{i} : {x_clustered_dict[i].shape}, part from all labels: {100*num_for_label/num_all_labels:.2f}%")
#     return x_clustered_dict, score, idxs_dict


# +
def get_kmeans_elbow_scores(x,max_n_clusters=20,verbose=1,mode="sklearn_minibatch"):
    """
        x: should be normalized per channel with normalize_channels_averages and flattened(1), shape: [N,C*H*W]
        returns an np.array of average of all standard deviations per value of each cluster  
        see get_kmeans_clusters_dict for available modes
    """
    clusters_scores = []
    for n_clusters in tqdm(range(1,max_n_clusters+1)):
        print(f"# Calculating for {n_clusters} clusters...")
        clusters_dict,score,_ = get_kmeans_clusters_dict(x,x,n_clusters,verbose=verbose,mode=mode)
        clusters_scores+=[score]
        print(f"Score: {clusters_scores[-1]:.2f}")
    clusters_scores = np.array(clusters_scores)
    if mode=="sklearn_minibatch":
        plot_title = "Averages of K-Means Clusters Standard Deviations"
        y_label = "Average Sum of Clusters Standard Deviation"
    else:
        plot_title = "Score of K-Means Clusters"
        y_label = "-(Score of Clusters)"
    plt.clf();
    fig,axes=plt.subplots(1,1,figsize=(9,9),dpi=70);
    axes.title.set_text(plot_title)
    axes.plot(np.arange(1,max_n_clusters+1),clusters_scores)
    axes.set_xlabel("Number of Clusters")    
    axes.set_ylabel(y_label) 
    plt.show()
    return clusters_scores

# def get_kmeans_elbow_stds(x,max_n_clusters=20,verbose=1,mode="sklearn_minibatch"):
#     """
#         x: should be normalized per channel with normalize_channels_averages and flattened(1), shape: [N,C*H*W]
#         returns an np.array of average of all standard deviations per value of each cluster  
#     """
#     clusters_stds = []
#     for n_clusters in tqdm(range(1,max_n_clusters+1)):
#         print(f"Calculating for {n_clusters} clusters...")
#         clusters_dict = get_kmeans_clusters_dict(x,x,n_clusters,verbose=verbose)
#         clusters_stds+=[sum([clusters_dict[i].std(0).sum() for i in range(n_clusters)])/n_clusters]
#         print(f"Sum of standard deviation of clusters: {clusters_stds[-1]:.2f}")
#     clusters_stds = np.array(clusters_stds)
#     title = "Averages of K-Means Clusters' Standard Deviations"
#     plt.clf();
#     fig,axes=plt.subplots(1,1,figsize=(9,9),dpi=70);
#     axes.title.set_text(title)
#     axes.plot(np.arange(1,max_n_clusters+1),clusters_stds)
#     axes.set_xlabel("Number of Clusters")    
#     axes.set_ylabel("Average Sum of Clusters' Standard Deviation") 
#     plt.show()
#     return clusters_stds

# def get_kmeans_elbow_scores_sklearn(x,max_n_clusters=20,verbose=1,speed="fast",mode="sklearn_minibatch"):
#     """
#         x: should be normalized per channel with normalize_channels_averages and flattened(1), shape: [N,C*H*W]
#         returns an np.array of average of all standard deviations per value of each cluster  
#     """
#     clusters_scores = []
#     for n_clusters in tqdm(range(1,max_n_clusters+1)):
#         print(f"Calculating for {n_clusters} clusters...")
#         clusters_dict,score,_ = get_kmeans_clusters_dict_sklearn(x,x,n_clusters,verbose=verbose,speed=speed)
#         clusters_scores+=[score]
#         print(f"Sum of standard deviation of clusters: {clusters_scores[-1]:.2f}")
#     clusters_scores = np.array(clusters_scores)
#     title = "Averages of K-Means Clusters' Standard Deviations"
#     plt.clf();
#     fig,axes=plt.subplots(1,1,figsize=(9,9),dpi=70);
#     axes.title.set_text(title)
#     axes.plot(np.arange(1,max_n_clusters+1),clusters_scores)
#     axes.set_xlabel("Number of Clusters")    
#     axes.set_ylabel("Average Sum of Clusters' Standard Deviation") 
#     plt.show()
#     return clusters_scores


# +
def get_patched_tensor(t, patch_size,flatten_patches=False):
    """
        example: t shape = [5, 6, 81, 189] with kernel size = (27,27) -> out shape = [5, 21, 4374]
        if not flatten_patches: [5, 6, 21, 729] (5,6,num_patches,patch_size)
    """
    t_unf_flatten = torch.nn.functional.unfold(t,kernel_size=patch_size, dilation=1, padding=0, stride=patch_size).transpose(1,2)
    if flatten_patches:
        return t_unf_flatten
    C = t.shape[1]
    N,num_patches,patch_size = t_unf_flatten.shape
    return t_unf_flatten.reshape([N,num_patches,C,patch_size//C]).transpose(1,2)

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


# +
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


# -
def average_related_channels(t,channels_list):
    channels_averages = []
    for channels in tqdm(channels_list):
        t_channels = t[:,channels,:,:]
        channels_averages.append(t_channels.mean(1))
    channels_averages = torch.stack(channels_averages,dim=1)
    print(channels_averages.shape)
    return channels_averages


# +
def get_sift(t,n_features=0,contrastThreshold=0.04,edgeThreshold=10,to_plot=False):
    """
        t num of dims = 2
    """
    img = t.detach().cpu().numpy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features,contrastThreshold=contrastThreshold,
                                       edgeThreshold=edgeThreshold)
    keypoints, descriptors = sift.detectAndCompute(img,None)
    if to_plot:
        gray = cv2.cvtColor(np.stack([img]*3,axis=2), cv2.COLOR_BGR2GRAY)
        img_to_plot = cv2.drawKeypoints(gray,keypoints,img)
        plt.imshow(img_to_plot)        
        plt.show()
    keypoints_xy = np.array([k.pt for k in keypoints])
    return keypoints_xy,descriptors

def get_compressed_sift_moments(t,n_features=0,contrastThreshold=0.04,edgeThreshold=10,
                                weight_descriptors_by_descriptors_means=False,
                                weight_keypoints_by_descriptors_means=False):
    """
        t: shape of [N,C,H,W]
        output: shape: [N,C,8], f 
        Consists of: for each row and for each channel, the list of n_features positions' moments of 
        SIFT keypoints, and their 4 moments of their descriptors (mean, std, skews and kurtosis).
        If weight_by_descriptors_means: each moment is weighted by the descriptors' means (mean of 128 values 
        for each descriptor)
        weight_keypoints_by_descriptors_means: the moments of keypoints' xy will be weighted by their descriptors'
        means
    """
    def get_moments(t2d):
        if t2d is None or len(t2d)==0:
            return torch.zeros([1,4])
        means = t2d.mean(axis=1)
        diffs = t2d - means[:,None]
        stds = t2d.std(axis=1)
        zscores = diffs/stds[:,None]
        skews = (np.power(zscores, 3)).mean(axis=1)
        kurtosis = (np.power(zscores, 4)).mean(axis=1)
        moments = np.stack([means,stds,skews,kurtosis],axis=1)
        return moments
    all_keypoints_xy = []
    all_descriptors = []
    N,C,H,W = t.shape
    for i in tqdm(range(N)):
        moments_keypoints_xy_all_channels = []
        moments_descriptors_all_channels = []
        for c in range(C):
            tensor = t[i,c]
            keypoints_xy,descriptors = get_sift(t[i,c],n_features=n_features,to_plot=False,
                                                contrastThreshold=contrastThreshold,
                                                edgeThreshold=edgeThreshold) # [n,2],[n,128]
            moments_keypoints_xy_channel = get_moments(keypoints_xy)
            moments_descriptors_channel = get_moments(descriptors)
            if weight_keypoints_by_descriptors_means:
                moments_keypoints_xy_channel*=moments_descriptors_channel[:,0][:,None]
            if weight_descriptors_by_descriptors_means:
                moments_descriptors_channel*=moments_descriptors_channel[:,0][:,None]
            moments_keypoints_xy_all_channels.append(moments_keypoints_xy_channel.mean(axis=0))
            moments_descriptors_all_channels.append(moments_descriptors_channel.mean(axis=0))
        all_keypoints_xy.append(np.stack(moments_keypoints_xy_all_channels,axis=0))
        all_descriptors.append(np.stack(moments_descriptors_all_channels,axis=0))
    all_keypoints_xy = torch.tensor(np.stack(all_keypoints_xy,axis=0)).to(dtype=torch.float32)
    all_descriptors = torch.tensor(np.stack(all_descriptors,axis=0)).to(dtype=torch.float32)
    return torch.cat([all_keypoints_xy,all_descriptors],dim=2)

def get_min_num_of_sift_features(t,contrastThreshold=0.04,edgeThreshold=10):
    """
        t shape: [N,H,W]
    """
    min_num_of_features = 10000
    for i in range(t.shape[0]):
        keypoints_xy,_ = get_sift(t[i],n_features=0,contrastThreshold=contrastThreshold,
                                  edgeThreshold=edgeThreshold,to_plot=False)
        num_features = keypoints_xy.shape[0]
        if num_features<min_num_of_features:
            min_num_of_features=num_features
    return min_num_of_features

def get_pca_compression(t,n_components):
    # t is of shape [N,features] - returns [N,n_components]
    pca_reducer = torchdrift.reducers.PCAReducer(n_components=n_components)
    return pca_reducer.fit(t)

def get_all_sifts_per_channel(t,verbose=1,contrastThreshold=0.04,edgeThreshold=10,num_features=0):
    """
        t shape: [N,C,H,W]
    """
    N,C,H,W = t.shape
    min_num_of_features_channels = [num_features]*C
    if num_features==0:
        min_num_of_features_channels = []
        if verbose>0:
            print("Calculating minimal number of features per channel...")
        for c in range(C):
            min_num_of_features_channels.append(get_min_num_of_sift_features(t[:,c,:,:],
                                                                             contrastThreshold=contrastThreshold,
                                                                             edgeThreshold=edgeThreshold))
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
            xy,descriptors = get_sift(t[i,c],n_features=n_features,to_plot=False,
                                      contrastThreshold=contrastThreshold,edgeThreshold=edgeThreshold)
            xy=xy[:n_features]
            descriptors=descriptors[:n_features]
            xy_channel.append(torch.tensor(xy))
            descriptors_channel.append(torch.tensor(descriptors))
        sift_xy_channels.append(torch.stack(xy_channel,dim=0))
        sift_descriptors_channels.append(torch.stack(descriptors_channel,dim=0))
    if verbose>0:
        print(f"... Done! Results shapes: xy: {len(sift_xy_channels)}, " \
              f"descriptors: {len(sift_descriptors_channels)}")
        for c in range(C):
            print(f"Channel {c}: xy: {sift_xy_channels[c].shape}, "\
                  f"descriptors: {sift_descriptors_channels[c].shape}")    
    return sift_xy_channels,sift_descriptors_channels

def get_sift_pca_compressed(t,n_hists,n_descriptors,to_normalize=True,contrastThreshold=0.04,edgeThreshold=10):
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
    sift_xy_channels,sift_descriptors_channels = get_all_sifts_per_channel(t,contrastThreshold=contrastThreshold,
                                                                           edgeThreshold=edgeThreshold)
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
        desc_i = [get_pca_compression(sift_descriptors_channels[c][i],n_hists) for c in range(C)] #C*[3~50,n_hists]
        desc_i = torch.stack([get_pca_compression(desc_i[c].transpose(0,1),n_descriptors).transpose(0,1).float()
                             for c in range(C)],dim=0) #[C,n_descriptors,n_hists] ([6,3,10])
        t_descriptors.append(desc_i)
    t_descriptors = torch.stack(t_descriptors,dim=0) #[N,C,n_descriptors,n_hists]
    print(f"... Done! {t_descriptors.shape}")
    out = torch.cat([t_xy,t_descriptors],dim=3)
    print(f"Result shape: {out.shape}")
    return out

def get_sift_pca_compressed_simple(t,n_descriptors, n_components_xy, n_components_descriptors,
                                   to_normalize=True,contrastThreshold=0.04,edgeThreshold=10):
    """
        t shape: N,C,H,W 
        Returns the n_descriptors-PCA compression of the (SIFT features concatenated with their positions)
        Assuming all channels will have the same num of descriptors
        i.e: 
        input: t [N,C,H,W] -> [N,C,n_descriptors,2+128] -> 
        flattened: [N,C*n_descriptors*130] -> [N,n_components_xy+n_components_descriptors]
        if n_components_xy==0 and n_components_descriptors==0: no compression
        n_descriptors==0: will use all the descriptors found per channel
    """
    N,C,H,W = t.shape
    if to_normalize:
        print(f"Normalizing t...")
        t = (t-t.mean([0,2,3])[None,:,None,None])/t.std([0,2,3])[None,:,None,None]
        print("... Done!")
    print(f"Calculating SIFT features...")
    sift_xy_channels,sift_descriptors_channels = get_all_sifts_per_channel(t,contrastThreshold=contrastThreshold,
                                                                           edgeThreshold=edgeThreshold,
                                                                           num_features=n_descriptors)
    # C*[N,n_descriptors,2] and C*[N,n_descriptors,128]
    t_xy = torch.cat([sift_xy_channels[c].flatten(1).float() for c in range(C)],dim=1) #[N,C*n_descriptors*2]
    t_desc = torch.cat([sift_descriptors_channels[c].flatten(1).float() for c in range(C)],dim=1) #[N,C*n_descriptors*128]
    if n_components_xy==0 and n_components_descriptors==0:
        return torch.cat([t_xy,t_desc],dim=1)
    t_xy_pca = get_pca_compression(t_xy,n_components_xy) #[N,n_components_xy]
    t_desc_pca = get_pca_compression(t_desc,n_components_descriptors) #[N,n_components_descriptors]
    return torch.cat([t_xy_pca,t_desc_pca],dim=1) #[N,n_components_xy+n_components_descriptors]
# -









