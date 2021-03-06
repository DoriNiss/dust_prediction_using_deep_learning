# used for printing a meteorological parameter, no normalization assumed

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import math


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
        
def get_num_rows_known_cols(num_tensors,num_cols):
    return num_tensors//num_cols
    
def get_num_cols_known_rows(num_tensors,num_rows):
    return num_tensors//num_rows
    
def get_plot_parameters(num_rows, num_cols):
    if num_cols==4:
        fig_w = 15
    if num_cols==5:
        fig_w = 17
    if num_cols==6:
        fig_w = 19
    if num_cols==7:
        fig_w = 21
    if num_rows==2:
        cb_0,cb_h,fig_h = 0.21,0.58,5
    if num_rows==3:
        cb_0,cb_h,fig_h = 0.15,0.7,7
    if num_rows==4:
        cb_0,cb_h,fig_h = 0.15,0.7,9
    if num_rows==5:
        cb_0,cb_h,fig_h = 0.15,0.7,11
    if num_rows==6:
        cb_0,cb_h,fig_h = 0.15,0.7,13
    if num_rows==7:
        cb_0,cb_h,fig_h = 0.15,0.7,15
    if num_rows==8:
        cb_0,cb_h,fig_h = 0.15,0.7,17
    figsize = (fig_w,fig_h)
    return cb_0,cb_h,figsize

def print_tensors_with_cartopy(tensors, main_title="", titles=None, num_rows=None, num_cols=None,
                               lons=None, lats=None, save_as="",lock_bar=False, lock_bar_idxs=None, 
                               num_levels=None, levels_around_zero=False, manual_levels=None,
                               titles_only_on_edges=False, cols_titles=None, rows_titles=None,
                               lock_bar_rows_separately=False):
    """
        tensors: list of tensors of shape [len(lats),len(lons)]
    """
    lons = lons or [n/2. for n in range(-44*2,50*2+1)]
    lats = lats or [n/2. for n in range(20*2,60*2+1)]
    num_tensors = len(tensors)
    titles = titles or [""]*num_tensors
    if num_tensors != len(titles):
        print("Error! Bad length of tensors or titles")
        return
    if len(lons) != tensors[0].shape[-1] or len(lats) != tensors[0].shape[-2]:
        print(f"Error! Bad length of lons,lats ({len(lons),len(lats)}) and tensor's shape ({tensors[0].shape})")
        return
    plt.clf();
    projection=ccrs.PlateCarree()
    if lock_bar:
        lock_bar_idxs = lock_bar_idxs or list(range(len(tensors)))
        if manual_levels is None:
            stacked_t =torch.stack([tensors[idx] for idx in lock_bar_idxs])
            vmin,vmax = stacked_t.min(),stacked_t.max()
            if levels_around_zero:
                extreme = math.ceil(max(abs(vmin),abs(vmax)))*1.
                vmin,vmax = -extreme,extreme
            step_size = ((vmax-vmin)/num_levels)*1.00001
            levels = [i*step_size+vmin for i in range(num_levels+1)]
        else:
            levels = manual_levels
    num_cols = num_cols or 4 # default value
    num_rows = num_rows or get_num_rows_known_cols(num_tensors,num_cols)
    cb_0,cb_h,figsize = get_plot_parameters(num_rows, num_cols)
    fig,axes=plt.subplots(num_rows,num_cols,figsize=figsize,dpi=100,subplot_kw={'projection': projection});
    plt.set_cmap('bwr')
    main_title_h = 0.92 if lock_bar else 0.95
    fig.text(0.4, main_title_h, main_title, horizontalalignment='center', verticalalignment='top',fontdict={"fontsize":18})
    fig.subplots_adjust(wspace=0.5, hspace=0.7)
    for row in range(num_rows):
        if lock_bar_rows_separately:
            tensors_idxs = [row*num_cols+col for col in range(num_cols)]
            stacked_t = torch.stack([tensors[idx] for idx in tensors_idxs])
            vmin,vmax = stacked_t.min(),stacked_t.max()
            if levels_around_zero:
                extreme = math.ceil(max(abs(vmin),abs(vmax)))*1.
                vmin,vmax = -extreme,extreme
            step_size = ((vmax-vmin)/num_levels)*1.00001
            levels = [i*step_size+vmin for i in range(num_levels+1)]
        for col in range(num_cols):
            i = row*num_cols+col
            t = tensors[i]
            t = t.detach().cpu()
            axes[row,col].set_extent([lons[0], lons[-1], lats[0], lats[-1]], crs=ccrs.PlateCarree())
            axes[row,col].coastlines(resolution='110m',lw=0.6)
            if (lock_bar and i in lock_bar_idxs) or lock_bar_rows_separately:
                c=axes[row,col].contourf(lons,lats,t,levels=levels)
            else:
                c=axes[row,col].contourf(lons,lats,t)
                clb=plt.colorbar(c, shrink=0.5, pad=0.05, ax=axes[row,col])  
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            if titles_only_on_edges:
                if row==0:
                    axes[row,col].title.set_text(cols_titles[col])
                if col==0:
                    axes[row,col].text(-0.08, 0.5, rows_titles[row],
                                       horizontalalignment='center',
                                       verticalalignment='center',
                                       rotation='vertical',
                                       transform=axes[row,col].transAxes)
            else:
                axes[row,col].title.set_text(titles[i])
        if lock_bar_rows_separately:
            cax = axes[row,col].inset_axes([1.05, 0, 0.02, 1], transform=axes[row,col].transAxes)
            plt.colorbar(mappable=c, cax=cax, shrink=0.5, pad=0.05) 
    if lock_bar:
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes([0.82, cb_0, 0.01, cb_h])
        plt.colorbar(mappable=c, cax=cax, shrink=0.5, pad=0.05)  
    plt.show();  
    if save_as != "":
        plt.savefig(save_as)        
        print("Saved to: ", save_as)  
    return 

def print_channel_samples(tensors,main_title="",titles=None,channels=None,num_rows=None,num_cols=None,lons=None, 
                          lats=None, save_as="",lock_bar=False,lock_bar_idxs=None,num_levels=10,
                          levels_around_zero=False, manual_levels=None):
    channels = channels or [0,1,4,5,8,9,13,14]
    tensors = tensors if len(tensors)>=8 else [tensors[0][c] for c in channels] # assuming shape [C,H,W]
    print_tensors_with_cartopy(tensors, main_title=main_title,titles=titles,num_rows=num_rows,num_cols=num_cols,
                               lons=lons,lats=lats,save_as=save_as,lock_bar=lock_bar,lock_bar_idxs=lock_bar_idxs, 
                               num_levels=num_levels,levels_around_zero=levels_around_zero,manual_levels=manual_levels)
