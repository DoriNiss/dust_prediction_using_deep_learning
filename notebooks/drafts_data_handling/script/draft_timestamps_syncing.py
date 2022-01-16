#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from tqdm.notebook import tqdm


def sync_timestamps(timestamps_lists,verbose=0):
    """ Return a list of lists of idxs to take from each list"""
    shared_timestamps = timestamps_lists[0]
    if len(shared_timestamps)>1:
        for timestamps_list in timestamps_lists[1:]:
            shared_timestamps = shared_timestamps.intersection(timestamps_list)
    if verbose>0: print(f"Found shared timestamps, {len(shared_timestamps)}. Finding indices: ...")
    shared_timestamps_idxs = []
    for timestamps_list in tqdm(timestamps_lists):
        shared_idxs_per_list = []
        for shared_t in shared_timestamps:
            try:
                shared_idxs_per_list.append(timestamps_list.get_loc(shared_t))
            except:
                print(f"Error! Something wierd happend with {shared_t} and {timestamps_list}. Aborting")
                return
        shared_timestamps_idxs.append(shared_idxs_per_list)
    if verbose>0: print(f"Done! Checking lengths of lists: ...")
    length_of_result = len(shared_timestamps)
    for idxs_list in shared_timestamps_idxs:
        if len(idxs_list)!=length_of_result:
            print(f"Error! Bad length of timestamps list: {idxs_list}. Aborting")
            return
    if verbose>0: print(f"Done!")
    return shared_timestamps_idxs, shared_timestamps


timestamps1 = pd.to_datetime([f"01-0{i}-2000" for i in range(1,10)])
timestamps2 = pd.to_datetime([f"01-0{i}-2000" for i in range(5,10)]+[f"01-{i}-2000" for i in range(10,15)])
timestamps3 = pd.to_datetime([f"01-0{i}-2000" for i in range(4,8)])
timestamps_lists = [timestamps1,timestamps2,timestamps3]
timestamps_lists





shared_timestamps_idxs, shared_timestamps = sync_timestamps(timestamps_lists,verbose=1)





# Checking:


shared_timestamps_idxs


shared_timestamps


print(shared_timestamps)
for list_idx in range(len(timestamps_lists)):
    print(pd.to_datetime([timestamps_lists[list_idx][i] for i in shared_timestamps_idxs[list_idx]]))










