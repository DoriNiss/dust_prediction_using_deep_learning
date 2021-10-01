#!/usr/bin/env python
# coding: utf-8

import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate 

import sys
sys.path.insert(0, '../../packages/')
from data_handlers.DustPredictionDataset import *
from utils.meteorology_printing import *
from training.train_model import *


debug_tensor_path = "../../data/tensors_debug_1/tensor_train_meteorology.pkl"
t = torch.load(debug_tensor_path)
t.shape


print_parameter(t[3]*0.95+t[20]*0.05,5) 
print_parameter(t[3],5) 
print_parameter(t[20],5) 








noise_set = t[:4,:,:,:]
noise_set.shape


noise_odds = 0.4 # = N/(N+Z)
N = noise_set.shape[0] # N,C,H,W
num_zeros = int(N/noise_odds-N)
N, num_zeros, (N/(N+num_zeros))


N,C,H,W = noise_set.shape
N,C


noise_set_zeros_expanded = torch.cat((noise_set.new_zeros(num_zeros,noise_set.shape[1],noise_set.shape[2],noise_set.shape[3]),noise_set))
noise_set_zeros_expanded.shape


torch.arange(10.)


idxs_to_choose_from = torch.tensor([i for i in range(noise_set_zeros_expanded.shape[0])])*1.
idxs_to_choose_from


choosen_idxs = torch.multinomial(idxs_to_choose_from,num_samples=t.shape[0],replacement=True)

choosen_idxs.shape, choosen_idxs, noise_set_zeros_expanded[choosen_idxs].shape


# !pip install git+https://github.com/pvigier/perlin-numpy


from perlin_numpy import generate_perlin_noise_2d

print("Perlin")
# np.random.seed(0)
noise = generate_perlin_noise_2d((81, 81), (3, 3))
# noise = (noise*(noise>=0))
plt.imshow(noise, cmap='gray', interpolation='lanczos')
plt.colorbar()
plt.show()

noise_idx = 11
sample_idx = 20
param_idx = 5
print("Noise-added")
print_parameter(0.95*t[sample_idx]+t[noise_idx]*0.5*noise,param_idx)
print("Noise-free")
print_parameter(t[sample_idx],param_idx)
print("Noise tensor")
print_parameter(t[noise_idx],param_idx)
print("Difference")
print_parameter(t[noise_idx]-t[sample_idx],param_idx)


debug_dir = "../../data/tensors_debug_1/"
debug_meteorology_train_path = debug_dir+"tensor_train_meteorology.pkl"
debug_dust_train_path = debug_dir+"tensor_train_dust.pkl"

metadata_dir = "../../data/metadata_meteo20000101to20210630_dust_0_m24_24_48_72/"
metadata_times_split1_train_path = metadata_dir+"split1_ordered_train_times.pkl"
times_debug = torch.load(metadata_times_split1_train_path)[:48]

debug_tensor_full = torch.load(debug_meteorology_train_path)
debug_dust_full = torch.load(debug_dust_train_path)

debug_tensor_full.shape, debug_dust_full.shape


from data_handlers.augmentations import *
from torch.utils.data import Dataset,DataLoader

perlin_augmentation = PerlinAugmentation(debug_tensor_full[10:11,:,:,:], debug_dust_full[10:11,:], debug=True)
print(debug_dust_full[10:11,0])


times_debug[0:1]


debug_dust_full[0:1,0]


dataset_augmented = DustPredictionDataset(debug_tensor_full[0:2,:,:,:],debug_dust_full[0:2,:],times_debug[0:2],
                                                augmentation=perlin_augmentation)

dataset_not_augmented = DustPredictionDataset(debug_tensor_full[0:2,:,:,:],debug_dust_full[0:2,:],times_debug[0:2])

data_loader_augmented = DataLoader(dataset_augmented, batch_size=64, shuffle=True,collate_fn=dataset_augmented.collate_fn)
data_loader_not_augmented = DataLoader(dataset_not_augmented, batch_size=64, shuffle=True,collate_fn=dataset_not_augmented.collate_fn)

sample_data_augmented = next(iter(data_loader_augmented))
print(sample_data_augmented[0][0].shape, sample_data_augmented[0][1].shape, len(sample_data_augmented[1]))

sample_data_not_augmented = next(iter(data_loader_not_augmented))
print(sample_data_not_augmented[0][0].shape, sample_data_not_augmented[0][1].shape, len(sample_data_not_augmented[1]))

t_augmented = sample_data_augmented[0][0]
t_not_augmented = sample_data_not_augmented[0][0]
print(t_augmented.shape,t_not_augmented.shape)


param_idx = 14
print_parameter(t_augmented[0],param_idx)
print_parameter(t_not_augmented[0],param_idx)
print_parameter(t_augmented[0]-t_not_augmented[0],param_idx)











# some experiments with try and exceptions


a = torch.tensor([0])
a


a.shape


# torch.multinomial(a*1.,num_samples=1,replacement=True)


try:
    print(torch.multinomial(a*1.,num_samples=1,replacement=True))
except Exception as exp:
    print("Nope")
    print(exp)


a = torch.tensor([[1.],[1.]])
b = torch.tensor([[2.],[2.]])
torch.cat([a,b],1)











# Loss debug


from training.dust_loss import *


weights_lags = [2,2,2,2,2]


loss_cfg = LossConfig("cpu", lags_indices=None, delta_lags_indices=None, weights_lags=weights_lags, weights_delta_lags=None)


dust_pred = torch.ones([1,10])
dust_target = torch.zeros([1,10])
dust_pred


dust_loss(dust_pred, dust_target, loss_cfg)


# correct
# decaying weights:


np.log(np.exp(1))


# exp(t=10) = 0.1 = e^(-R*t) => R = -log(0.1)/10
r = -np.log(0.1)/10
np.exp(-r*10)


weights_lags_list = [1., np.exp(-r*1), np.exp(-r*1), np.exp(-r*2), np.exp(-r*3)]
weights_lags_list


[weights_lags_list[i]/2 for i in range(len(weights_lags_list))]




