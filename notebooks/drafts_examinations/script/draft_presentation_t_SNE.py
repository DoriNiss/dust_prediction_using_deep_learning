#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../packages/')

import torch
import matplotlib.pyplot as plt

from openTSNE import TSNE


# !pip install opentsne





data_dir = "../../data/tensors_meteo20000101to20210630_dust_0_m24_24_48_72/presentation_set/"

presentation_meteorology_train_path = data_dir+"tensor_train_meteorology.pkl"
presentation_meteorology_valid_path = data_dir+"tensor_valid_meteorology.pkl"
presentation_dust_train_path = data_dir+"tensor_train_dust.pkl"
presentation_dust_valid_path = data_dir+"tensor_valid_dust.pkl"
presentation_times_train_path = data_dir+"times_train.pkl"
presentation_times_valid_path = data_dir+"times_valid.pkl"


presentation_meteorology_valid = torch.load(presentation_meteorology_valid_path)


presentation_meteorology_valid.shape, presentation_meteorology_valid.flatten(1).shape


tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)


get_ipython().run_line_magic('time', 'embedding_valid = tsne.fit(presentation_meteorology_valid.flatten(1))')


plt.scatter(embedding_valid[:,0],embedding_valid[:,1])


# scatter?


embedding_valid.shape


plt.scatter(embedding_valid[:,0],embedding_valid[:,1])





results_dir = "../../results_models/presentation/"


torch.save(embedding_valid,results_dir+"tSNE_valid_presentation.pkl")


presentation_meteorology_train = torch.load(presentation_meteorology_train_path)
get_ipython().run_line_magic('time', 'embedding_train = tsne.fit(presentation_meteorology_train[:6000].flatten(1))')


presentation_meteorology_train.shape, presentation_meteorology_train.flatten(1).shape


plt.scatter(embedding_train[:,0],embedding_train[:,1])


torch.save(embedding_train,results_dir+"tSNE_train_presentation_6000.pkl")








# sequences - valid year
presentation_meteorology_sequential_valid_path = data_dir+"meteorology_sequential_0m12243648_valid.pkl"
presentation_dust_sequential_valid_path = data_dir+"dust_sequential_0m12243648_valid.pkl"
presentation_times_sequential_valid_path = data_dir+"times_sequential_0m12243648_valid.pkl"


presentation_meteorology_valid.shape, presentation_meteorology_valid.flatten(1).shape


presentation_meteorology_valid_sequence = torch.load(presentation_meteorology_sequential_valid_path)
get_ipython().run_line_magic('time', 'embedding_valid_sequence = tsne.fit(presentation_meteorology_valid_sequence.flatten(1))')


plt.scatter(embedding_valid_sequence[:,0],embedding_valid_sequence[:,1])


torch.save(embedding_valid_sequence,results_dir+"tSNE_sequence_valid_presentation.pkl")


presentation_meteorology_valid_sequence.shape





# sequences - 2017+2018 year
presentation_meteorology_sequential_train_path = data_dir+"meteorology_sequential_0m12243648_train_2018.pkl"
presentation_dust_sequential_train_path = data_dir+"dust_sequential_0m12243648_train_2018.pkl"
presentation_times_sequential_train_path = data_dir+"times_sequential_0m12243648_train_2018.pkl"


a = 2
a


presentation_meteorology_sequential_train = torch.load(results_dir+"presentation_meteorology_sequential_train_path")


presentation_meteorology_sequential_2017_2018_0m12243648 = torch.cat((presentation_meteorology_valid_sequence,
                                                                      presentation_meteorology_sequential_train),0)
presentation_meteorology_sequential_2017_2018_0m12243648.shape


get_ipython().run_line_magic('time', 'embedding_sequential_2017_2018 = tsne.fit(presentation_meteorology_sequential_2017_2018_0m12243648.flatten(1))')
torch.save(embedding_sequential_2017_2018,results_dir+"tSNE_sequence_2017_2018_0m12243648_presentation.pkl")
plt.scatter(embedding_sequential_2017_2018[:,0],embedding_sequential_2017_2018[:,1])




