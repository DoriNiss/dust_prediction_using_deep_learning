#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np


t = torch.zeros([8,1])
t


t[2,0] = 1
t[4,0] = 2
t


r = 0.5
num_events = int(t.shape[0]*r)
num_events


populate_from = (t>0).nonzero() # as_tuple=True
populate_from


to_populate = (t==0).nonzero()
to_populate


num_to_populate = num_events-(t>0).count_nonzero()
num_to_populate


# newly_populated_idxs = torch.multinomial(populate_from*1., num_samples=num_to_populate.detach().item(), replacement=False)
newly_populated_idxs = torch.multinomial(populate_from[:,0]*1., 10, replacement=True)

populate_from[newly_populated_idxs]


newly_populated_idxs = to_populate[torch.multinomial(to_populate[:,0]*1., 
                                                     num_samples=num_to_populate.detach().item(), 
                                                     replacement=True)]
newly_populated_idxs


a = torch.zeros([batch_size,10])
a[2,:] = 10.
a[8,:] = 20.
(a[:,0]>0).nonzero()[:,0]




















batch_size = 10


meteorology = torch.ones([batch_size,3,2,2])
for i in range(meteorology.shape[0]):
    meteorology[i] *= i


dust = torch.zeros([batch_size])
dust[2] = 10
dust[5] = 20
dust


events_idxs = (dust>0).nonzero()[:,0]
clear_idxs = (dust==0).nonzero()[:,0]
events_idxs, clear_idxs


meteorology[events_idxs], dust[events_idxs]


events_idxs.shape[0]


r = 0.5
final_num_events = int(r*batch_size)
num_events_to_populate = final_num_events-events_idxs.shape[0]
final_num_events,num_events_to_populate


events_to_populate_idxs = events_idxs[torch.multinomial(events_idxs*1.,
                                                        num_samples=num_events_to_populate,
                                                        replacement=True)]
events_to_populate = dust[events_to_populate_idxs]
events_to_populate_idxs,events_to_populate


clear_to_populate_idxs = clear_idxs[torch.multinomial(clear_idxs*1.,
                                                      num_samples=num_events_to_populate,
                                                      replacement=True)]
clear_to_populate = dust[clear_to_populate_idxs]
clear_to_populate_idxs,clear_to_populate


dust[clear_to_populate_idxs] = events_to_populate
print("tensor([ 0.,  0., 10.,  0.,  0., 20.,  0.,  0.,  0.,  0.])")
dust


meteorology[clear_to_populate_idxs] = meteorology[events_to_populate_idxs]
meteorology




