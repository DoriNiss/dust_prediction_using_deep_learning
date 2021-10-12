# Currently aimed for one map as output

import sys
sys.path.insert(0, '../../../packages/')
from utils.metrics import *
from utils.training_loop_plotting import *
from utils.meteorology_printing import *
from training.dataset_handlers.MapsToMapsDataset_NoAugmentation import *

import numpy as np
import copy
from tqdm import tqdm

def train_epoch(model, optimizer, loader, device, criterion, debug=False, scheduler=None):
    loss_metric = Metric()
    for inputs,targets,_ in loader:
        inputs,targets = inputs.to(device=device), targets.to(device=device)
        optimizer.zero_grad()
        model.train()
        outputs = model(inputs)
        loss = criterion(outputs,targets) 
        loss.backward()
        loss_metric.update(loss.item(), inputs.size(0))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return loss_metric

def valid_epoch(model, loader, device, criterion):
    loss_metric = Metric()
    for inputs,targets,_ in loader:
        inputs,targets = inputs.to(device=device), targets.to(device=device)
        model.eval()
        outputs = model(inputs)
        loss = criterion(outputs,targets) 
        loss_metric.update(loss.item(), inputs.size(0))
    return loss_metric

def train_loop(model, optimizer, train_loader, valid_loader, device, epochs, criterion, valid_every=1,
               sample_outputs_every=20, sample_size=5, loss_plot_end=True, debug=False, 
               save_to_dir=None, scheduler=None):
    print("Training... \n\n")
    train_losses = []
    valid_losses = []
    best_valid_loss = 1e20
    best_model_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, optimizer, train_loader, device, criterion, debug=debug, scheduler=scheduler)
        train_losses.append(train_loss.avg)
        print('Train', f'Epoch: {epoch:03d} / {epochs:03d}', f'Loss: {train_loss.avg:7.4g}', sep='   ')
        if epoch % valid_every == 0:
            valid_loss =  valid_epoch(model, valid_loader, device, criterion)
            valid_losses.append(valid_loss.avg)
            print('Valid', f'Epoch: {epoch:03d} / {epochs:03d}', f'Loss: {valid_loss.avg:7.4g}', sep='   ')
        if epoch % sample_outputs_every == 0 or epoch==epochs:
            sample_inputs, sample_targets, _ = next(iter(train_loader)) # can add timestamps to plots
            sample_inputs, sample_targets = sample_inputs.to(device), sample_targets.to(device)
            sample_outputs = model(sample_inputs)
            print_2_parameters_no_cartopy([sample_targets[0],sample_outputs[0]] , 0, main_title="Sample Result",
                                          titles=["Sample Target","Sample Output"])
        if loss_plot_end and epoch==epochs:
            plot_train_valid(train_losses,valid_losses)
        if valid_loss.avg<best_valid_loss:
            best_valid_loss = valid_loss.avg
            best_model_state = copy.deepcopy(model.state_dict())
            if save_to_dir is not None: # TODO: add checkpoint here (so no jupyter will be needed)
                torch.save(model.state_dict(),save_to_dir+"best.pkl")
        if save_to_dir is not None:
            torch.save(model.state_dict(),save_to_dir+"last.pkl")
            torch.save(train_losses,save_to_dir+"train_losses.pkl")
            torch.save(valid_losses,save_to_dir+"valid_losses.pkl")

    model.load_state_dict(best_model_state)
    return (train_losses,valid_losses)