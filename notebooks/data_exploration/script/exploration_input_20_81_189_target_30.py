#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/')
from utils.files_loading import *

import matplotlib.pyplot as plt


years_list = list(range(2003,2005))
data_dir = "../../data/datasets_20_81_189_averaged_dust_24h"
base_filename = "dataset_20_81_189_averaged_dust_24h"


inputs,targets,timestamps = load_stacked_inputs_targets_timestamps_from_years_list(years_list,data_dir,base_filename)


a = torch.ones([2,4,6])
a[1]*=10
a[:,:,1]+=1
a[:,:,2]+=2
a[:,:,2]-=1
a[:,-1,:]+=3
a[:,1,:]+=1
plt.imshow(a[0])





import 
from utils.files_loading import *











a = torch.ones([2,4,6])
a[1]*=10
a[:,:,1]+=1
a[:,:,2]+=2
a[:,:,2]-=1
a[:,-1,:]+=3
a[:,1,:]+=1
plt.imshow(a[0])


a


t = torch.ones([2,3])
t/=t.sum()
t


t_new = t.unsqueeze(0).repeat(2,1,1)
t_new.shape
t_new.unsqueeze(0).shape


# torch.nn.functional.conv2d(a,t_new)


# t = torch.nn.Unfold([2,3])
# t(a.unsqueeze(0))


a.shape


a_unf.transpose(1, 2)


t_new


t_new.view(t_new.size(0), -1).t()


a


out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))


t.shape
w = t.unsqueeze(0).unsqueeze(0)
w = w.repeat(2,1,1,1)
w.shape, a_new.shape


conved = torch.nn.functional.conv2d(a_new, w, bias=None, stride=(2,3), padding=0, dilation=1, groups=2)
conved, conved.shape


(10+11+11+11+12+12)/6


add_labels(new_target,thresholds=thresholds,labels=labels,label_by_col=0)

