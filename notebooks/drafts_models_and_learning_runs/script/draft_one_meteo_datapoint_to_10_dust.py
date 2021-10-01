#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
import timm.models.vision_transformer as ViT
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate 
from tqdm import tqdm


# !pip install timm


# utils:
class Metric:
    def __init__(self):
        self.lst = 0.
        self.sum = 0.
        self.cnt = 0
        self.avg = 0.
    def update(self, val, cnt=1):
        self.lst = val
        self.sum += val * cnt
        self.cnt += cnt
        self.avg = self.sum / self.cnt

def tp_fp_fn_batch(pred, target, th=73.4, dust_0_idx=0):
    target_dust = target[:,dust_0_idx]
    pred_dust = pred[:,dust_0_idx]
    tp = ((pred_dust>=th)&(target_dust>=th)).count_nonzero() # predicted event and was right
    fp = ((pred_dust>=th)&(target_dust<th)).count_nonzero() # predicted event and was wrong
    fn = ((pred_dust<th)&(target_dust>=th)).count_nonzero() # predicted clear and was wrong
    return tp,fp,fn

def metrics_to_precision_recall(tp_metric, fp_metric, fn_metric):
    precision = (tp_metric.sum/(tp_metric.sum+fp_metric.sum)).item()
    if np.isnan(precision): precision=0
    recall = (tp_metric.sum/(tp_metric.sum+fn_metric.sum)).item()
    return precision,recall
    


# import numpy as np
# a = np.array([np.nan,np.nan])
# a[~np.isnan(a)],  a[~np.isnan(a)].size == 0


# pred = torch.tensor([100,0,0,100,100]).unsqueeze(0).t()
# targ = torch.tensor([0,100,0,0,100]).unsqueeze(0).t()
# precision_recall(pred,targ)


# paths

debug_dir = "../../data/tensors_debug_1/"
debug_meteorology_train_path = debug_dir+"tensor_train_meteorology.pkl"
debug_meteorology_valid_path = debug_dir+"tensor_valid_meteorology.pkl"
debug_dust_train_path = debug_dir+"tensor_train_dust.pkl"
debug_dust_valid_path = debug_dir+"tensor_valid_dust.pkl"

data_dir = "../../data/tensors_meteo20000101to20210630_dust_0_m24_24_48_72/"
split1_dir = data_dir+"split1_ordered/"
split2_dir = data_dir+"split2_extreme_ratios_in_train_avg_in_valid/"
split3_dir = data_dir+"split3_extreme_ratios_in_valid_avg_in_train/"
split4_dir = data_dir+"split4_max_num_events_in_train_avg_valid/"
split5_dir = data_dir+"split5_train_distant_years_valid_between/"
split_dirs = [split1_dir,split2_dir,split3_dir,split4_dir,split5_dir]
meteorology_train_paths = [split+"tensor_train_meteorology.pkl" for split in split_dirs]
meteorology_valid_paths = [split+"tensor_valid_meteorology.pkl" for split in split_dirs]
dust_train_paths = [split+"tensor_train_dust.pkl" for split in split_dirs]
dust_valid_paths = [split+"tensor_valid_dust.pkl" for split in split_dirs]

metadata_dir = "../../data/metadata_meteo20000101to20210630_dust_0_m24_24_48_72/"
metadata_columns_path = metadata_dir+"all_columns.pkl"
metadata_yearly_statistics_path = metadata_dir+"yearly_statistics.pkl"
metadata_all_times_path = metadata_dir+"all_times.pkl"
metadata_times_split1_train_path = metadata_dir+"split1_ordered_train_times.pkl"
metadata_times_split2_train_path = metadata_dir+"split2_extreme_ratios_in_train_avg_in_valid_train_times.pkl"
metadata_times_split3_train_path = metadata_dir+"split3_extreme_ratios_in_valid_avg_in_train_train_times.pkl"
metadata_times_split4_train_path = metadata_dir+"split4_max_num_events_in_train_avg_valid_train_times.pkl"
metadata_times_split5_train_path = metadata_dir+"split5_train_distant_years_valid_between_train_times.pkl"
metadata_times_train_paths = [metadata_times_split1_train_path,metadata_times_split2_train_path,
                              metadata_times_split3_train_path,metadata_times_split4_train_path,
                              metadata_times_split5_train_path]
metadata_times_split1_valid_path = metadata_dir+"split1_ordered_valid_times.pkl"
metadata_times_split2_valid_path = metadata_dir+"split2_extreme_ratios_in_train_avg_in_valid_valid_times.pkl"
metadata_times_split3_valid_path = metadata_dir+"split3_extreme_ratios_in_valid_avg_in_train_valid_times.pkl"
metadata_times_split4_valid_path = metadata_dir+"split4_max_num_events_in_train_avg_valid_valid_times.pkl"
metadata_times_split5_valid_path = metadata_dir+"split5_train_distant_years_valid_between_valid_times.pkl"
metadata_times_valid_paths = [metadata_times_split1_valid_path,metadata_times_split2_valid_path,
                              metadata_times_split3_valid_path,metadata_times_split4_valid_path,
                              metadata_times_split5_valid_path]


# TBD: get times of debug dataset
# import sys
# sys.path.insert(0, '../../packages/data_handlers')
# from Dataset_handler import *
# debug_dataset_handler = torch.load(debug_dir+"dummy_dataset_handler.pkl") # not working
times_debug = torch.load(metadata_times_split1_valid_path)[:48] # incorrect times - to be corrected
len(times_debug)


torch.load(metadata_columns_path)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# metadata: indices are 0:'dust_0',   1:'delta_0', 2:'dust_m24', 3:'delta_m24', 4:'dust_24',
#                       5:'delta_24', 6:dust_48',  7:'delta_48', 8:'dust_72',   9:'delta_72']
default_lags_indices = [0,2,4,6,8]
default_delta_lags_indices = [1,3,5,7,9]
default_lag_weights = torch.tensor([1.,1.,1.,1.,1.], device=device).double()
default_loss_cfg = {
    "lags_indices": default_lags_indices,
    "delta_lags_indices": default_delta_lags_indices,
    "lag_weights": default_lag_weights,
    "delta_lag_weights": default_lag_weights,
}


# loss
def dust_loss(dust_pred, dust_target, loss, loss_cfg=default_loss_cfg):
    # weights are tensors of the same device as the dust tensors
    loss_lags,loss_delta_lags = 0,0
    lags_pred = dust_pred[:,loss_cfg["lags_indices"]]
    delta_lags_pred = dust_pred[:,loss_cfg["delta_lags_indices"]]
    lags_target = dust_target[:,loss_cfg["lags_indices"]]
    delta_lags_target = dust_target[:,loss_cfg["delta_lags_indices"]]
    weights_lags = torch.sqrt(loss_cfg["lag_weights"])
    weights_delta_lags = torch.sqrt(loss_cfg["delta_lag_weights"])
    loss_lags = loss(lags_pred*weights_lags,lags_target*weights_lags)
    loss_delta_lags = loss(delta_lags_pred*weights_delta_lags,delta_lags_target*weights_delta_lags)
    return loss_lags + loss_delta_lags


# dust_tensor = torch.tensor([10.,1.,20.,2.,30.,3.,40.,4.,50.,5.])
# dust_tensor = torch.cat([dust_tensor.unsqueeze(0),2*dust_tensor.unsqueeze(0)],0)
# print(dust_tensor)
# print("loss:")
# dust_loss(dust_tensor, torch.zeros_like(dust_tensor), nn.MSELoss()) # 858.5


def plot_train_valid(train_losses, valid_losses):
    if len(train_losses) is not len(valid_losses):
        print("Wrong lengths - could not plot")
        return
    x = [i for i in range(len(train_losses))]
    fig, ax = plt.subplots()
    ax.plot(x, train_losses, label='training loss')
    ax.plot(x, valid_losses, label='validation loss')
    legend = ax.legend(loc='upper right')
    plt.show()    


def timestamp_collate(batch):
    new_batch = []
    timestamps = []
    for _batch in batch:
        new_batch.append(_batch[:-1])
        timestamps.append(_batch[-1])
    return default_collate(new_batch), timestamps

class DustPredictionDataset(Dataset):
    def __init__(self, meteorology_full_tensor, dust_full_tensor, times):
        self.meteorology = meteorology_full_tensor
        self.dust = dust_full_tensor
        self.times = times

    def __len__(self):
        return self.dust.shape[0]

    def __getitem__(self, idx):
        return self.meteorology[idx,:,:,:], self.dust[idx,:], self.times[idx]


def train_epoch(model, criterion, optimizer, loader, device, loss_cfg=default_loss_cfg):
    loss_metric = Metric()
    tp_metric = Metric()
    fp_metric = Metric()
    fn_metric = Metric()
    for minibatch, _ in loader:
        x=minibatch[0]
        y=minibatch[1].double()
        x, y = x.to(device=device), y.to(device=device)
        optimizer.zero_grad()
        model.train()
        pred = model(x)
        loss = dust_loss(pred, y, criterion, loss_cfg=loss_cfg)
        loss.backward()
        loss_metric.update(loss.item(), x.size(0))
        optimizer.step()
        tp,fp,fn = tp_fp_fn_batch(pred,y)
        tp_metric.update(tp, 1)
        fp_metric.update(fp, 1)
        fn_metric.update(fn, 1)
    precision,recall = metrics_to_precision_recall(tp_metric,fp_metric,fn_metric)
    return loss_metric,precision,recall

def valid_epoch(model, criterion, loader, device, loss_cfg=default_loss_cfg):
    loss_metric = Metric()
    tp_metric = Metric()
    fp_metric = Metric()
    fn_metric = Metric()
    for minibatch, _ in loader:
        x=minibatch[0]
        y=minibatch[1].double()
        x, y = x.to(device=device), y.to(device=device)
        model.eval()
        pred = model(x)
        loss = dust_loss(pred, y, criterion, loss_cfg=loss_cfg)
        loss_metric.update(loss.item(), x.size(0))
        tp,fp,fn = tp_fp_fn_batch(pred,y)
        tp_metric.update(tp, 1)
        fp_metric.update(fp, 1)
        fn_metric.update(fn, 1)
    precision,recall = metrics_to_precision_recall(tp_metric,fp_metric,fn_metric)
    return loss_metric,precision,recall

def train_loop(model, criterion, optimizer, train_loader, valid_loader, device, 
               epochs, valid_every=1,loss_cfg=default_loss_cfg):
    print("Training... (Precision = out all of predicted events, <> were correct, Recall = out of all events, predicted <>)\n\n")
    train_losses = []
    valid_losses = []
    for epoch in range(1, epochs + 1):
        train_loss,train_prec,train_recall = train_epoch(model, criterion, optimizer, train_loader, device, loss_cfg=loss_cfg)
        train_losses.append(train_loss.avg)
        print('Train', f'Epoch: {epoch:03d} / {epochs:03d}',
              f'Loss: {train_loss.avg:7.4g}',
              f'Precision: {train_prec*100:.3f}%',
              f'Recall: {train_recall*100:.3f}%',
              sep='   ')
        if epoch % valid_every == 0:
            valid_loss,valid_prec,valid_recall =  valid_epoch(model, criterion, valid_loader, device, loss_cfg=loss_cfg)
            valid_losses.append(valid_loss.avg)
            print('Valid',
                  f'                Loss: {valid_loss.avg:7.4g}',
                  f'Precision: {valid_prec*100:.3f}%',
                  f'Recall: {valid_recall*100:.3f}%',
                sep='   ')
    return train_losses,valid_losses








# debug
train_dataset = DustPredictionDataset(torch.load(debug_meteorology_train_path),
                                      torch.load(debug_dust_train_path),times_debug)
valid_dataset = DustPredictionDataset(torch.load(debug_meteorology_valid_path),
                                      torch.load(debug_dust_valid_path),times_debug)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=timestamp_collate)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=timestamp_collate)

model = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, depth=8,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
model = model.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 8
debug_train_losses,debug_valid_losses = train_loop(model, criterion, optimizer, train_loader, valid_loader, device, 
               epochs=num_epochs, valid_every=1,loss_cfg=default_loss_cfg)

plot_train_valid(debug_train_losses,debug_valid_losses)

print("Sample predictions:")
sample_data = next(iter(valid_loader))
print("Targets:")
print(sample_data[0][1][:4])
print("Predictions:")
print(model(sample_data[0][0][:4,:,:,:].to(device)))


sample_data = next(iter(train_loader))
sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1])





# split1
train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[0]),
                                      torch.load(dust_train_paths[0]),
                                      torch.load(metadata_times_train_paths[0]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[0]),
                                      torch.load(dust_valid_paths[0]),
                                      torch.load(metadata_times_valid_paths[0]))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=timestamp_collate)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=timestamp_collate)

sample_data = next(iter(train_loader))
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))


model_split1 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
model_split1 = model_split1.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.01
optimizer = torch.optim.Adam(model_split1.parameters(), lr=lr)
num_epochs = 80

train_losses_split1,valid_losses_split1 = train_loop(model_split1, criterion, optimizer, train_loader, valid_loader, 
                                          device, epochs=num_epochs, valid_every=1,loss_cfg=default_loss_cfg)

plot_train_valid(train_losses_split1,valid_losses_split1)

print("Sample predictions:")
sample_data = next(iter(valid_loader))
print("Targets:")
print(sample_data[0][1][:4])
print("Predictions:")
print(model_split1(sample_data[0][0][:4,:,:,:].to(device)))





# split2
train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[1]),
                                      torch.load(dust_train_paths[1]),
                                      torch.load(metadata_times_train_paths[1]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[1]),
                                      torch.load(dust_valid_paths[1]),
                                      torch.load(metadata_times_valid_paths[1]))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=timestamp_collate)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=timestamp_collate)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

model_split2 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
model_split2 = model_split2.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.1
optimizer = torch.optim.Adam(model_split2.parameters(), lr=lr)
num_epochs = 80

train_losses_split2,valid_losses_split2 = train_loop(model_split2, criterion, optimizer, train_loader, valid_loader, 
                                          device, epochs=num_epochs, valid_every=1,loss_cfg=default_loss_cfg)

plot_train_valid(train_losses_split2,valid_losses_split2)

print("Sample predictions:")
sample_data = next(iter(valid_loader))
print("Targets:")
print(sample_data[0][1][:4])
print("Predictions:")
print(model_split2(sample_data[0][0][:4,:,:,:].to(device)))





# split3
train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[2]),
                                      torch.load(dust_train_paths[2]),
                                      torch.load(metadata_times_train_paths[2]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[2]),
                                      torch.load(dust_valid_paths[2]),
                                      torch.load(metadata_times_valid_paths[2]))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=timestamp_collate)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=timestamp_collate)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

model_split3 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
model_split3 = model_split3.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.01
optimizer = torch.optim.Adam(model_split3.parameters(), lr=lr)
num_epochs = 80

train_losses_split3,valid_losses_split3 = train_loop(model_split3, criterion, optimizer, train_loader, valid_loader, 
                                          device, epochs=num_epochs, valid_every=1,loss_cfg=default_loss_cfg)

plot_train_valid(train_losses_split3,valid_losses_split3)

print("Sample predictions:")
sample_data = next(iter(valid_loader))
print("Targets:")
print(sample_data[0][1][:4])
print("Predictions:")
print(model_split3(sample_data[0][0][:4,:,:,:].to(device)))





# split4
train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[3]),
                                      torch.load(dust_train_paths[3]),
                                      torch.load(metadata_times_train_paths[3]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[3]),
                                      torch.load(dust_valid_paths[3]),
                                      torch.load(metadata_times_valid_paths[3]))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=timestamp_collate)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=timestamp_collate)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

model_split4 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
model_split4 = model_split4.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.001
optimizer = torch.optim.Adam(model_split4.parameters(), lr=lr)
num_epochs = 80

train_losses_split4,valid_losses_split4 = train_loop(model_split4, criterion, optimizer, train_loader, valid_loader, 
                                          device, epochs=num_epochs, valid_every=1,loss_cfg=default_loss_cfg)

plot_train_valid(train_losses_split4,valid_losses_split4)

print("Sample predictions:")
sample_data = next(iter(valid_loader))
print("Targets:")
print(sample_data[0][1][:4])
print("Predictions:")
print(model_split4(sample_data[0][0][:4,:,:,:].to(device)))





# split5
train_dataset = DustPredictionDataset(torch.load(meteorology_train_paths[4]),
                                      torch.load(dust_train_paths[4]),
                                      torch.load(metadata_times_train_paths[4]))
valid_dataset = DustPredictionDataset(torch.load(meteorology_valid_paths[4]),
                                      torch.load(dust_valid_paths[4]),
                                      torch.load(metadata_times_valid_paths[4]))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=timestamp_collate)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,collate_fn=timestamp_collate)

sample_data = next(iter(train_loader))
print("Sample data loading:")
print(sample_data[0][0].shape, sample_data[0][1].shape, len(sample_data[1]))

model_split5 = ViT.VisionTransformer(img_size=(81,81), patch_size=(9,9), in_chans=17, num_classes=10, embed_dim=512, 
                 depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
model_split5 = model_split5.to(device)

criterion = nn.MSELoss() # to be used inside the dust_loss
lr = 0.0001
optimizer = torch.optim.Adam(model_split5.parameters(), lr=lr)
num_epochs = 80

train_losses_split5,valid_losses_split5 = train_loop(model_split5, criterion, optimizer, train_loader, valid_loader, 
                                          device, epochs=num_epochs, valid_every=1,loss_cfg=default_loss_cfg)

plot_train_valid(train_losses_split5,valid_losses_split5)

print("Sample predictions:")
sample_data = next(iter(valid_loader))
print("Targets:")
print(sample_data[0][1][:4])
print("Predictions:")
print(model_split5(sample_data[0][0][:4,:,:,:].to(device)))


meteo_tensor = torch.load(meteorology_train_paths[4])


meteo_tensor.shape


torch.load(meteorology_valid_paths[4]).shape


def show_tensor(self, t, param=7, rand=False, show_3=False, denorm=False, lock_scale=False, title="", 
                    save_as="",quality=50):
        plt.clf();    
        projection=ccrs.PlateCarree()
        fig,axes=plt.subplots(1,1,figsize=(7,9),dpi=quality,subplot_kw={'projection': projection});
        shrink=0.4
        plt.set_cmap('bwr')
        lons = np.array([i for i in range(-30,61)])
        lats = np.array([i for i in range(0,71)])
        arrow_scale = 400 if denorm else 10
        title_str = "" if title == "" else ", "+title
        #fig.suptitle(date_str)
        axes.title.set_text(param_title+title_str)
        axes.set_extent([-30, 60, 0, 70], crs=ccrs.PlateCarree())
        axes.coastlines(resolution='110m',lw=0.6)
        if type(lock_scale) != bool or (type(lock_scale) == bool and lock_scale):
            c=axes.contourf(lons,lats, data, levels=levels, extend='both')
        else:
            c=axes.contourf(lons,lats, data)
        if show_3:
            axes.quiver(lons[::quiver_s[idx]],lats[::quiver_s[idx]],u[::quiver_s[idx],::quiver_s[idx]], 
                   v[::quiver_s[idx],::quiver_s[idx]],scale=arrow_scale)
        clb=plt.colorbar(c, shrink=shrink, pad=0.05, ax=axes)   
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])    
        plt.show();  
        if save_as != "":
            plt.savefig(save_as)        
            print("Saved to: ", save_as)





# TDL:
# 1. Plot data
# 2. Put into package
#    Create new heavier debug dataset
# 3. Importance sampling
# 4. Augmentation (read, add gaussian noise?)
# 4. WEXAC: a. learn how to send batch jobs, b. remote desktop / anydesk + linux from here




