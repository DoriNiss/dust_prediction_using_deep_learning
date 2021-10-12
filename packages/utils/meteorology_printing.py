# used for printing a meteorological parameter, no normalization assumed

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# +
def print_parameter(full_tensor, p_idx, title="", lons=None,lats=None, save_as=""):
    # Assuming temsor of shape [c,lats,longs] (should be given an instance from the batch, e.g. t[8])
    import cartopy.crs as ccrs # install with: pip install cartopy    
    if lons is None:
        lons = [n/2. for n in range(0*2,40*2+1)]
    if lats is None:
        lats = [n/2. for n in range(20*2,60*2+1)]
    plt.clf();
    projection=ccrs.PlateCarree()
    fig,axes=plt.subplots(1,1,figsize=(7,9),dpi=50,subplot_kw={'projection': projection});
    plt.set_cmap('bwr')
    axes.title.set_text(title)
    axes.set_extent([lons[0], lons[-1], lats[0], lats[-1]], crs=ccrs.PlateCarree())
    axes.coastlines(resolution='110m',lw=0.6)
    c=axes.contourf(lons,lats, full_tensor[p_idx])
    clb=plt.colorbar(c, shrink=0.4, pad=0.05, ax=axes)  
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show();  
    if save_as != "":
        plt.savefig(save_as)        
        print("Saved to: ", save_as)
        
def print_2_parameters_no_cartopy(tensors, channel, main_title="", titles=["",""],
                                  inverse_rows=True, lons=None, lats=None, save_as=""):
    # Assuming temsor of shape [c,lats,longs]
    plt.clf();
    lons = lons or [n/2. for n in range(-44*2,40*2+1)]
    lats = lats or [n/2. for n in range(20*2,60*2+1)]
    fig,axes=plt.subplots(1,2,figsize=(13,4),dpi=50);
    plt.set_cmap('bwr')
    fig.text(0.5, 1.1, main_title, horizontalalignment='center', verticalalignment='top',
                fontdict={"fontsize":18})
    fig.subplots_adjust(wspace=0.5, hspace=0.7)
    for i in range(0,2):
        axes[i].title.set_text(titles[i])
        t = tensors[i][channel].flip(0) if inverse_rows else tensors[i][channel]
        t = t.detach().cpu()
        c=axes[i].contourf(lons,lats, t)
        clb=plt.colorbar(c, shrink=0.5, pad=0.05, ax=axes[i])  
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show();  
    if save_as != "":
        plt.savefig(save_as)        
        print("Saved to: ", save_as)
