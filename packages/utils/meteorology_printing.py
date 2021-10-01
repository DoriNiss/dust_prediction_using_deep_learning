# used for printing a meteorological parameter, no normalization assumed

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs # install with: pip install cartopy

def print_parameter(full_tensor, p_idx, title="", lons=None,lats=None, save_as=""):
    # Assuming temsor of shape [c,lats,longs] (should be given an instance from the batch, e.g. t[8])
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
