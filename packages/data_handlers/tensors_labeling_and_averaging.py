import torch
import numpy as np


def add_labels(x, thresholds=[73.4], labels=[0,1], label_by_col=0):
    """
        Assuming x of shape: [N,cols]
        e.g.:
        thresholds = [4,10], labels = [0,1,2]: will add column with the following values (at position [:,-1]):
            0: x<4
            1: 4<=x<10
            2: x>=10
    """
    labels_as_tensors = torch.tensor(labels,device=x.device)*1.
    x_labels = torch.zeros([x.shape[0],1],device=x.device)+labels[0]
    for i,th in enumerate(thresholds):
        x_labels = torch.where((x[:,label_by_col]>=th).unsqueeze(0).transpose(1,0),labels_as_tensors[i+1],x_labels)
    x_labeled = torch.cat([x,x_labels],dim=1)
    print(f"Added labels {labels} at position [:,-1], new shape: {x_labeled.shape}")
    return x_labeled

def average_cols_and_drop_invalid(x,cols_to_average,valid_threshold,invalid_values=None):
    """
        Assuming x of shape: [N,cols]
        cols_to_average: list of numpy arrays representing the cols to be averaged, e.g.: [np.arange(0,4),np.arange(4,8)...]
        invalid_values: values to be considered when counting number of valid values
        valid_threshold: if encounteres less than valid_threshold[i]*len(cols_to_average[i]) valid values when averaging,
            drops the rows
        returns list of rows' indices to keep and a new tensor of averaged columns
    """
    invalid_values = invalid_values or [np.nan, np.inf, -np.inf]
    averages_list = []
    for cols,drop_rate in zip(cols_to_average,valid_threshold):
        x_cols = x[:,cols]
        cols_average_per_row = []
        min_num_of_valid = int(drop_rate*len(cols))
        for row in range(x_cols.shape[0]):
            valid_counter = 0
            for c in cols:
                if x[row,c] not in invalid_values: valid_counter+=1
            if valid_counter<min_num_of_valid:
                cols_average_per_row+=[torch.tensor(np.nan)]
            else:
                cols_average_per_row+=[x_cols[row].mean()]
        averages_list+=[torch.tensor(cols_average_per_row)]
    x_averaged = torch.stack(averages_list).transpose(0,1)
    indices_to_keep = torch.isnan(x_averaged).sum(dim=1)==0
    rows_to_keep = np.arange(x_averaged.shape[0])[indices_to_keep]
    return x_averaged, rows_to_keep


def duplicate_col_i(x,i):
    col_i = x[:,i].unsqueeze(0).transpose(0,1)
    x_left = x[:,:i]
    x_right = x[:,i:]
    return torch.cat([x_left,col_i,x_right],dim=1)


