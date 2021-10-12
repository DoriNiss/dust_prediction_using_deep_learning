import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import sys
sys.path.insert(0, '../../packages/')
from utils.training_loop_plotting import *


def print_all_losses(train_loss, valid_loss,
                     train_lags_losses, train_delta_lags_losses, 
                     valid_lags_losses, valid_delta_lags_losses, 
                     title="Prediction Losses", times=None):
    """
        train_loss / valid_loss - list of epochs' total loss during training / validation
        lags_losses - list of epochs' separated lags losses during
        delta_lags_losses - list of epochs' separated delta lags losses
    """
    times = times or ["T+0h","T-24h","T+24h","T+48h","T+72h"]
    fig = plt.figure(figsize=(15, 3), dpi=80)
    # fig.suptitle(title, fontsize=16)
    fig.text(0.5, 1.15, title, horizontalalignment='center', verticalalignment='top',
             fontdict={"fontsize":18})
    fig.subplots_adjust(wspace=0.5, hspace=0.7)
    gs = gridspec.GridSpec(2, 9)
    ax = fig.add_subplot(gs[:,:3])
    x = np.arange(len(train_loss))
    ax.plot(x, train_loss, label='Training')
    ax.plot(x, valid_loss, label='Validation')
    ax.legend(loc='upper right')    
    ax.set_title("Total Epoch Loss", fontdict={"fontsize":14})
    losses_train = [train_lags_losses, train_delta_lags_losses]
    losses_valid = [valid_lags_losses, valid_delta_lags_losses]
    losses_titles = ["$PM_{10}$","$\u0394PM_{10}$"]
    for col,time in enumerate(times):
        for row in range(2):
            ax = fig.add_subplot(gs[row,col+4])
            y_train, y_valid = losses_train[row][:,col], losses_valid[row][:,col]
            x = np.arange(y_train.shape[0])
            ax.ticklabel_format(axis="y",style="sci",useMathText=True,scilimits=(0,0))
            ax.plot(x, y_train)
            ax.plot(x, y_valid)
            if row==0:
                ax.set_title(time,fontdict={"fontsize":14}, y=1.2)
            if col==0:
                ax.set_ylabel(losses_titles[row],fontdict={"fontsize":14})
    plt.show()

def print_predictions(predicted_dust, true_dust, times=None, idxs=None, spacing=50, 
                      show_times=True, bbox_to_anchor = (1.02,1.02), num_ticks=8, th=73.4):

    # assuming period is longer than 24
    idx_shifts_lags = {
        "$T+0h, PM_{10}$":  {"col_idx": 0, "idx_shift":  0}, 
        "$T-24h, PM_{10}$": {"col_idx": 2, "idx_shift":  8}, 
        "$T+24h, PM_{10}$": {"col_idx": 4, "idx_shift": -8}, 
        "$T+48h, PM_{10}$": {"col_idx": 6, "idx_shift":-16}, 
        "$T+72h, PM_{10}$": {"col_idx": 8, "idx_shift":-24}
    }
    idx_shifts_delta_lags = {
        "$T+0h, \u0394PM_{10}$":  {"col_idx": 1, "idx_shift":  0}, 
        "$T-24h, \u0394PM_{10}$": {"col_idx": 3, "idx_shift":  8}, 
        "$T+24h, \u0394PM_{10}$": {"col_idx": 5, "idx_shift": -8}, 
        "$T+48h, \u0394PM_{10}$": {"col_idx": 7, "idx_shift":-16}, 
        "$T+72h, \u0394PM_{10}$": {"col_idx": 9, "idx_shift":-24}, 
    }
    if idxs is None:
        idxs = np.arange(predicted_dust.shape[0])
    idx_start = idxs[0]+24
    idx_end = idxs[-1]-24
    # idxs_cut = np.arange(idx_start,idx_end)
    x = np.arange(idx_start,idx_end)
    time_ticks = times[x]
    time_ticks = time_ticks[:-1:(x.shape[0]//num_ticks)]
    x_for_xticks = x[:-1:(x.shape[0]//num_ticks)]
    fig = plt.figure(figsize=(8, 4), dpi=80)
    fig.text(0.5, 1.15, "Dust Predictions", horizontalalignment='center', verticalalignment='top',
                fontdict={"fontsize":18})
    fig.subplots_adjust(wspace=0.5, hspace=0.7)
    gs = gridspec.GridSpec(1, 2)
    ax = fig.add_subplot(gs[0,0])
    ax.plot(x,true_dust[idx_start:idx_end,0], label="Ground Truth")
    ax.plot(x, x*0+th, color="red", linestyle="dashed", label="Event Threshold")
    for i,lag_name in enumerate(idx_shifts_lags.keys()):
        step_string = "" if (i+1)*spacing==0 else "+"+str((i+1)*spacing)
        idxs_shifted_start = idx_shifts_lags[lag_name]["idx_shift"]+idx_start
        idxs_shifted_end = idx_shifts_lags[lag_name]["idx_shift"]+idx_end
        col = idx_shifts_lags[lag_name]["col_idx"]
        ax.plot(x, (i+1)*spacing+predicted_dust[idxs_shifted_start:idxs_shifted_end,col], 
                 label=lag_name+step_string)
        if show_times:
            plt.xticks(x_for_xticks,time_ticks,rotation=90)
    plt.plot(x,true_dust[idx_start:idx_end,1], label="Ground Truth")
    ax = fig.add_subplot(gs[0,1])
    for i,lag_name in enumerate(idx_shifts_delta_lags.keys()):
        step_string = "" if (i+1)*spacing==0 else "+"+str((i+1)*spacing)
        idxs_shifted_start = idx_shifts_delta_lags[lag_name]["idx_shift"]+idx_start
        idxs_shifted_end = idx_shifts_delta_lags[lag_name]["idx_shift"]+idx_end
        col = idx_shifts_delta_lags[lag_name]["col_idx"]
        plt.plot(x, (i+1)*spacing+predicted_dust[idxs_shifted_start:idxs_shifted_end,col], 
                 label=lag_name+step_string)
        plt.legend(bbox_to_anchor=bbox_to_anchor)
        if show_times:
            plt.xticks(x_for_xticks,time_ticks,rotation=90)
    plt.show()    

