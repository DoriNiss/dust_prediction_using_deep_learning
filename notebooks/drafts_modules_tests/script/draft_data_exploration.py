#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/')
from utils.files_loading import *
from utils.data_exploration import *
from utils.meteorology_printing import *

import cartopy.crs as ccrs

import matplotlib.pyplot as plt


# %load_ext autoreload
# %autoreload 2


# years_list = list(range(2003,2004))
# data_dir = "../../data/datasets_20_81_189_averaged_dust_24h"
# base_filename = "dataset_20_81_189_averaged_dust_24h"
# inputs,targets,timestamps = load_stacked_inputs_targets_timestamps_from_years_list(years_list,data_dir,base_filename)


# sample_idx = 45
# sample_channel = 10

# patch_sizes = [[1,1], [3,3], [9,9], [27,27], [9,21]]

# for patch_size in patch_sizes:
#     print(f"\n#### Patch_size: {patch_size}")
#     inputs_patch_averaged = patch_averages(inputs,patch_size)
#     print(f"Shape: {inputs_patch_averaged.shape}")
#     plt.imshow(inputs[sample_idx,sample_channel])
#     plt.show()
#     plt.imshow(inputs_patch_averaged[sample_idx,sample_channel])
#     plt.show()





# inputs_events,targets_events,timestamps_events,idxs_events = \
#     get_inputs_targets_timestamps_idxs_of_labels_only(inputs,targets,timestamps,1)





# !pip install fast-pytorch-kmeans


years_list = list(range(2003,2019))
data_dir = "../../data/datasets_20_81_189_averaged_dust_24h"
base_filename = "dataset_20_81_189_averaged_dust_24h"
description = torch.load(f"{data_dir}/metadata/dataset_20_81_189_averaged_dust_24h_metadata.pkl")


# inputs,targets,timestamps = load_stacked_inputs_targets_timestamps_from_years_list(years_list,data_dir,base_filename)


# torch.save(inputs,f"{data_dir}/{base_filename}_all_inputs.pkl")
# torch.save(targets,f"{data_dir}/{base_filename}_all_targets.pkl")
# torch.save(timestamps,f"{data_dir}/{base_filename}_all_timestamps.pkl")
inputs = torch.load(f"{data_dir}/{base_filename}_all_inputs.pkl")
targets = torch.load(f"{data_dir}/{base_filename}_all_targets.pkl")
timestamps = torch.load(f"{data_dir}/{base_filename}_all_timestamps.pkl")
inputs.shape,targets.shape,len(timestamps)


inputs_events,targets_events,timestamps_events,idxs_events =     get_inputs_targets_timestamps_idxs_of_labels_only(inputs,targets,timestamps,1)


inputs_clear,targets_clear,timestamps_clear,idxs_clear =     get_inputs_targets_timestamps_idxs_of_labels_only(inputs,targets,timestamps,0)








# kmeans


inputs_events_normed, events_means, events_stds = normalize_channels_averages(inputs_events)





# patch_size = [3,3]
# events_patched_averaged = patch_averages(inputs_events_normed,patch_size)
# print(events_patched_averaged.shape)
# x_kmeans = events_patched_averaged.flatten(1)
# print(x_kmeans.shape)


# kmeans_elbow_stds = get_kmeans_elbow_stds(x_kmeans,max_n_clusters=24,verbose=1)








patch_size = [81,189]
events_patched_averaged = patch_averages(inputs_events_normed,patch_size)
print(events_patched_averaged.shape)
x_kmeans = events_patched_averaged.flatten(1)
print(x_kmeans.shape)
kmeans_elbow_stds = get_kmeans_elbow_stds(x_kmeans,max_n_clusters=24,verbose=1)


# patch_size = [27,27]
# events_patched_averaged = patch_averages(inputs_events_normed,patch_size)
# print(events_patched_averaged.shape)
# x_kmeans = events_patched_averaged.flatten(1)
# print(x_kmeans.shape)
# kmeans_elbow_stds = get_kmeans_elbow_stds(x_kmeans,max_n_clusters=24,verbose=1)





clusters_dict = get_kmeans_clusters_dict(x_kmeans,inputs_events,4)


titles_channels = [description["input"][i]["short"] for i in range(20)]


# for c in range(20):
#     title = f"Channel: {titles_channels[c]}"
#     titles = [f"Events Cluster {i} Average" for i in range(4)]+[""]*16
#     tensors = [inputs_events[i,c] for i in range(4)]+[inputs_events[0,c]*0+inputs_events[0,c].min()]*16
#     print_tensors_with_cartopy(tensors, main_title=title, titles=titles,
#                                lons=None, lats=None, save_as="", lock_bar=True, num_levels=10)    
    


t_list = [torch.rand([2,3,4])]*20
t_stack = torch.stack(t_list)
t_stack.shape


t_stack.min(1)[0].min(1)[0].min(1)[0].shape





inputs_events_avgs,inputs_clear_avgs = [],[]

for i in tqdm(range(0,8)):
    inputs_events_i,targets_events_i,timestamps_events_i,idxs_events_i =         get_inputs_targets_timestamps_idxs_above_or_below_value(inputs,targets,timestamps,label_th=73.4,label_idx=i,
                                                                above_or_below="above")
    inputs_clear_i,targets_clear_i,timestamps_clear_i,idxs_clear_i =         get_inputs_targets_timestamps_idxs_above_or_below_value(inputs,targets,timestamps,label_th=73.4,label_idx=i,
                                                                above_or_below="below")
    inputs_events_avgs.append(batch_average_datapoint(inputs_events_i))
    inputs_clear_avgs.append(batch_average_datapoint(inputs_clear_i))


averages_all = []
titles_averages = []
for i in tqdm(range(8)):
    inputs_averages_i,targets_averages_i,timestamps_averages_i,idxs_averages_i =         get_inputs_targets_timestamps_idxs_above_or_below_value(inputs,targets,timestamps,label_th=10000,label_idx=i,
                                                                above_or_below="below")
    averages_all.append(batch_average_datapoint(inputs_averages_i))
    titles_averages+=[f"Average all: in {i} days"]
    


# empty_tensor_list = [inputs_events_avgs[i]*0 for i in range(4)]
# tensors_averages = empty_tensor_list+inputs_events_avgs+inputs_clear_avgs
# tensors_averages = [tensors_averages[i] for i in range(19,-1,-1)]

avgs_idxs = [7,4,2,0]

tensors_averages = [inputs_events_avgs[i]-averages_all[i] for i in range(7,-1,-1)] +                    [inputs_clear_avgs[i]-averages_all[i] for i in range(7,-1,-1)] +                    [averages_all[i] for i in avgs_idxs]
                            
titles = [f"Average Event - Average: in {i} days" for i in range(7,-1,-1)]+          [f"Average Clear - Average: in {i} days" for i in range(7,-1,-1)]+          [f"Average: in {i} days" for i in avgs_idxs]

for c in range(20):
    title = f"Channel: {titles_channels[c]}"
    tensors = [t[c] for t in tensors_averages]
    print_tensors_with_cartopy(tensors, main_title=title, titles=titles,
                               lons=None, lats=None, save_as="", lock_bar=True, lock_bar_idxs=list(range(16)),
                               num_levels=10)    





























len(tensors)


len(tensors_averages)


tensors_averages[0].shape


# (inputs_events-denormalize_channels_averages(inputs_events_normed,events_means,events_stds))
## close enough to 0





patch_size = [3,3]
events_patched_averaged = patch_averages(inputs_events_normed,patch_size)
events_patched_averaged.shape


x_kmeans = events_patched_averaged.flatten(1)
x_kmeans.shape


from fast_pytorch_kmeans import KMeans

num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, mode='euclidean', verbose=1)
events_labels = kmeans.fit_predict(x_kmeans)


len(events_labels)


plt.plot(events_labels)





events_by_label = {i: inputs_events[events_labels==i] for i in range(num_clusters)}


num_events = inputs_events.shape[0]
num_clears = inputs_clear.shape[0]
num_total = num_events+num_clears

for i in range(num_clusters):
    num_these_events = events_by_label[i].shape[0]
    print(f"{i} : {events_by_label[i].shape}, part from events: {100*num_these_events/num_events:.2f}%"           f", clears to these events: {num_these_events/num_clears:.2f}, part from all data: "          f"{100*num_these_events/num_total:.2f}%")





def get_kmeans_dict(x,num_clesters,verbose=1):
    """
        x: should be normalized per channel with normalize_channels_averages, shape: [N,C,H,W]
        returns a dict with {i: x[labels[i]}
    """
    kmeans = KMeans(n_clusters=num_clusters, mode='euclidean', verbose=verbose)
    labels = kmeans.fit_predict(x)
    labeled_x_dict = {i: x[labels==i] for i in range(num_clusters)}
    if verbose>0:
        num_all_labels = x.shape[0]
        for i in range(num_clusters):
            num_for_label = labeled_x_dict[i].shape[0]
            print(f"{i} : {labeled_x_dict[i].shape}, part from all labels: {100*num_for_label/num_all_labels:.2f}%")
    return labeled_x_dict


a = torch.ones([4,20])
a.std(0).shape


titles_channels = [description["input"][i]["short"] for i in range(20)]


for i in range(num_clusters):
    title = f"Average Event of Type {i}"
    print(f"\n{title}:")
    x = batch_average_datapoint(events_by_label[i])
    tensors = [x[c] for c in range(x.shape[0])]
    print_tensors_with_cartopy(tensors, main_title=title, titles=titles_channels,
                               lons=None, lats=None, save_as="")    


title = f"Average of Clear Days"
print(f"\n{title}:")
x = batch_average_datapoint(inputs_clear)
tensors = [x[c] for c in range(x.shape[0])]
print_tensors_with_cartopy(tensors, main_title=title, titles=titles_channels,
                           lons=None, lats=None, save_as="")   


# def print_tensors_with_cartopy(tensors, main_title="", titles=None,
#                                lons=None, lats=None, save_as=""):
#     """
#         tensors: list of tensors of shape [len(lats),len(lons)]
#     """
#     lons = lons or [n/2. for n in range(-44*2,50*2+1)]
#     lats = lats or [n/2. for n in range(20*2,60*2+1)]
#     num_tensors = len(tensors)
#     titles = titles or [""]*num_tensors
#     if num_tensors != len(titles):
#         print("Error! Bad length of tensors or titles")
#         return
#     if len(lons) != tensors[0].shape[-1] or len(lats) != tensors[0].shape[-2]:
#         print(f"Error! Bad length of lons,lats ({len(lons),len(lats)}) and tensor's shape ({tensors[0].shape})")
#         return
#     plt.clf();
#     projection=ccrs.PlateCarree()
#     if num_tensors==20:
#         num_rows,num_cols = 5,4
#         fig,axes=plt.subplots(num_rows,num_cols,figsize=(20,15),dpi=200,subplot_kw={'projection': projection});
#         plt.set_cmap('bwr')
#         fig.text(0.5, 1.1, main_title, horizontalalignment='center', verticalalignment='top',
#                     fontdict={"fontsize":18})
#         fig.subplots_adjust(wspace=0.5, hspace=0.7)
#         for row in range(num_rows):
#             for col in range(num_cols):
#                 i = row*num_cols+col
#                 axes[row,col].title.set_text(titles[i])
#                 t = tensors[i]
#                 t = t.detach().cpu()
#                 axes[row,col].set_extent([lons[0], lons[-1], lats[0], lats[-1]], crs=ccrs.PlateCarree())
#                 axes[row,col].coastlines(resolution='110m',lw=0.6)
#                 c=axes[row,col].contourf(lons,lats, t)
#                 clb=plt.colorbar(c, shrink=0.5, pad=0.05, ax=axes[row,col])  
#                 fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
#         plt.show();  
#         if save_as != "":
#             plt.savefig(save_as)        
#             print("Saved to: ", save_as)  
#         return  


# sample_t = inputs_events[0]
# 

# tensors = [sample_t[c] for c in range(sample_t.shape[0])]
# print_tensors_with_cartopy(tensors, main_title="", titles=titles_channels,
#                            lons=None, lats=None, save_as="")




















a = torch.ones([2,4,6])
a[1]*=10
a[:,:,1]+=1
a[:,:,2]+=2
a[:,:,2]-=1
a[:,-1,:]+=3
a[:,1,:]+=1
a_new=a.unsqueeze(0)
plt.imshow(a[0])
plt.show()
a_patch_averaged = patch_averages(a_new,[2,3])
a_patch_averaged, a_patch_averaged.shape




