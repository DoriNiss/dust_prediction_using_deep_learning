#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/')
from utils.files_loading import *
from data_handlers.tensors_labeling_and_averaging import *


import torch


a = torch.ones([5,8])
a[0,0]=0
a[0,5]=0
a[1,1]=0
a[1,2]=0
a[1,3]=0
a[1,4]=0
a[2,4]=0
a[2,6]=0
a[3,1]=0
a[3,2]=0
a[3,3]=0
a[4,5]=0
a[4,6]=0
a[4,7]=0

a


cols_to_average = [np.arange(4*i,4*(i+1)) for i in range(2)]
valid_threshold = [0.5]*2
cols_to_average,valid_threshold


average_cols_and_drop_invalid(a,cols_to_average,valid_threshold,invalid_values=[0])


# create new inputs and save
# create new metatdata and save


import numpy as np


a = np.arange(0,4)
len(a)







