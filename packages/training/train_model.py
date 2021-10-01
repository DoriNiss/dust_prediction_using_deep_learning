import sys
sys.path.insert(0, '../../packages/')
from utils.metrics import *
from utils.training_loop_plotting import *
from training.dust_loss import *
from data_handlers.DustPredictionDataset import *
import numpy as np
import copy

def train_epoch(model, optimizer, loader, device, loss_cfg, debug=False, scheduler=None):
    loss_metric = Metric()
    loss_lags_metric = Metric()
    loss_delta_lags_metric = Metric()
    tp_metric = Metric()
    fp_metric = Metric()
    fn_metric = Metric()
#     num_iters = len(loader)
    for minibatch, _ in loader:
        x=minibatch[0].float()
        y=minibatch[1].float()
        x,y = loader.dataset.sample_by_importance(x, y, debug=debug) # events_ratio=-1 will make it not sample_by_importance (won't change x,y)
        x = loader.dataset.augmentation.augment(x, y).float()
        x, y = x.to(device=device), y.to(device=device)
        optimizer.zero_grad()
        model.train()
        pred = model(x)
        loss,batch_loss_lags,batch_loss_delta_lags = dust_loss(pred, y, loss_cfg) 
        loss.backward()
        loss_metric.update(loss.item(), x.size(0))
        loss_lags_metric.update(batch_loss_lags.cpu().detach().numpy(), x.size(0))
        loss_delta_lags_metric.update(batch_loss_delta_lags.cpu().detach().numpy(), x.size(0))
        optimizer.step()
        if scheduler is not None:
            # assuming cosine scheduler...
#             scheduler.step(epoch + i/num_iters)
            scheduler.step()
#             print(scheduler.get_last_lr())
        tp,fp,fn = tp_fp_fn_batch(pred,y)
        tp_metric.update(tp, 1)
        fp_metric.update(fp, 1)
        fn_metric.update(fn, 1)
    precision,recall = metrics_to_precision_recall(tp_metric,fp_metric,fn_metric)
    return loss_metric,precision,recall,loss_lags_metric,loss_delta_lags_metric

def valid_epoch(model, loader, device, loss_cfg):
    loss_metric = Metric()
    loss_lags_metric = Metric()
    loss_delta_lags_metric = Metric()
    tp_metric = Metric()
    fp_metric = Metric()
    fn_metric = Metric()
    for minibatch, _ in loader:
        x=minibatch[0].float()
        y=minibatch[1].float()
        x, y = x.to(device=device), y.to(device=device)
        model.eval()
        pred = model(x)
        loss,batch_loss_lags,batch_loss_delta_lags = dust_loss(pred, y, loss_cfg) 
        loss_metric.update(loss.item(), x.size(0))
        loss_lags_metric.update(batch_loss_lags.cpu().detach().numpy(), x.size(0))
        loss_delta_lags_metric.update(batch_loss_delta_lags.cpu().detach().numpy(), x.size(0))
        tp,fp,fn = tp_fp_fn_batch(pred,y)
        tp_metric.update(tp, 1)
        fp_metric.update(fp, 1)
        fn_metric.update(fn, 1)
    precision,recall = metrics_to_precision_recall(tp_metric,fp_metric,fn_metric)
    return loss_metric,precision,recall,loss_lags_metric,loss_delta_lags_metric

def train_loop(model, optimizer, train_loader, valid_loader, device, epochs, valid_every=1,loss_cfg=None, 
               sample_predictions_every=2, sample_size=5, sample_cols=[0],
               loss_plot_end=True, debug=False, save_best_model_dict_to=None, save_last_model_dict_to=None,
               scheduler=None, losses_dir=None):
    print("Training... (Precision = out all of predicted events, <> were correct, Recall = out of all events, predicted <>)\n\n")
    train_losses = []
    train_lags_losses = []
    train_delta_lags_losses = []
    valid_losses = []
    valid_lags_losses = []
    valid_delta_lags_losses = []
    best_valid_loss = 1e20
    best_model_state = copy.deepcopy(model.state_dict())
    if loss_cfg is None: loss_cfg = LossConfig(device)
    for epoch in range(1, epochs + 1):
        train_loss,train_prec,train_recall,train_loss_lags,train_loss_delta_lags = train_epoch(model, optimizer, train_loader, device, loss_cfg, debug=debug, scheduler=scheduler)
        train_losses.append(train_loss.avg)
        train_lags_losses.append(train_loss_lags.avg)
        train_delta_lags_losses.append(train_loss_delta_lags.avg)
        print('Train', f'Epoch: {epoch:03d} / {epochs:03d}',
              f'Loss: {train_loss.avg:7.4g}',
              f'Precision: {train_prec*100:.3f}%',
              f'Recall: {train_recall*100:.3f}%',
              sep='   ')
        if epoch % valid_every == 0:
            valid_loss,valid_prec,valid_recall,valid_loss_lags,valid_loss_delta_lags =  valid_epoch(model, valid_loader, device, loss_cfg)
            valid_losses.append(valid_loss.avg)
            valid_lags_losses.append(valid_loss_lags.avg)
            valid_delta_lags_losses.append(valid_loss_delta_lags.avg)
            print('Valid',
                  f'                Loss: {valid_loss.avg:7.4g}',
                  f'Precision: {valid_prec*100:.3f}%',
                  f'Recall: {valid_recall*100:.3f}%',
                sep='   ')
        if epoch % sample_predictions_every == 0 or epoch==epochs:
            sample_data = next(iter(train_loader))
            sample_meteo, sample_targets = sample_data[0][0][0:sample_size], sample_data[0][1][0:sample_size]
            sample_meteo, sample_targets = sample_meteo.to(device), sample_targets.to(device)
            sample_predictions = model(sample_meteo)
            data_to_print = torch.cat([sample_predictions[:,sample_cols],sample_targets[:,sample_cols]],1).data.cpu().numpy()
            print(f"        [Sample predictions | targets] (cols [:{len(sample_cols)}] |  cols [{len(sample_cols)}:]):")
            print('\t'+str(data_to_print).replace('\n','\n\t'))
        if loss_plot_end and epoch==epochs:
            plot_train_valid(train_losses,valid_losses)
        if valid_loss.avg<best_valid_loss:
            best_valid_loss = valid_loss.avg
            best_model_state = copy.deepcopy(model.state_dict())
            if save_best_model_dict_to is not None: # TODO: add checkpoint here
                torch.save(model.state_dict(),save_best_model_dict_to)
        if save_last_model_dict_to is not None:
            torch.save(model.state_dict(),save_last_model_dict_to)
        if losses_dir is not None:
            torch.save(train_losses,losses_dir+"train_losses.pkl")
            torch.save(np.stack(train_lags_losses,axis=0),losses_dir+"train_lags_losses.pkl")
            torch.save(np.stack(train_delta_lags_losses,axis=0),losses_dir+"train_delta_lags_losses.pkl")
            torch.save(valid_losses,losses_dir+"valid_losses.pkl")
            torch.save(np.stack(valid_lags_losses,axis=0),losses_dir+"valid_lags_losses.pkl")
            torch.save(np.stack(valid_delta_lags_losses,axis=0),losses_dir+"valid_delta_lags_losses.pkl")

    model.load_state_dict(best_model_state)
    train_separated_lags_losses =  np.stack(train_lags_losses,axis=0)
    train_separated_delta_lags_losses =  np.stack(train_delta_lags_losses,axis=0)
    valid_separated_lags_losses =  np.stack(valid_lags_losses,axis=0)
    valid_separated_delta_lags_losses =  np.stack(valid_delta_lags_losses,axis=0)
    return (train_losses,valid_losses,
            train_separated_lags_losses,train_separated_delta_lags_losses,
            valid_separated_lags_losses,valid_separated_delta_lags_losses)
