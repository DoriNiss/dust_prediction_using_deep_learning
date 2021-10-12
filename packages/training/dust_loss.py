import torch
import torch.nn as nn
import numpy as np

def dust_loss(dust_pred, dust_target, loss_cfg):
    """
        returns: loss_final, batch-mean lags loss with shape of [1,5], same for delta lags
        loss_final is the lag-weighted MSE loss (sum of weighted MSE losses of lags and deltas)
    """
    loss_lags,loss_delta_lags = 0,0
    lags_pred = dust_pred[:,loss_cfg.lags_indices]
    delta_lags_pred = dust_pred[:,loss_cfg.delta_lags_indices]
    lags_target = dust_target[:,loss_cfg.lags_indices]
    delta_lags_target = dust_target[:,loss_cfg.delta_lags_indices]
    w = loss_cfg.weights_lags
    w_deltas = loss_cfg.weights_delta_lags
    loss_lags_fn = nn.MSELoss(reduction="none")
    loss_delta_lags_fn = nn.MSELoss(reduction="none")
    loss_lags = loss_lags_fn(lags_pred,lags_target)
    loss_delta_lags = loss_delta_lags_fn(delta_lags_pred,delta_lags_target)
    loss_final = (w*loss_lags).mean()+(w_deltas*loss_lags).mean()
    return loss_final, loss_lags.mean(0), loss_delta_lags.mean(0)

class LossConfig:
    def __init__(self, device, lags_indices=None, delta_lags_indices=None, weights_lags=None, weights_delta_lags=None, decaying_weights=False):
        """
            metadata: default indices are 0:'dust_0',   1:'delta_0', 2:'dust_m24', 3:'delta_m24', 4:'dust_24',
                                          5:'delta_24', 6:dust_48',  7:'delta_48', 8:'dust_72',   9:'delta_72']
        """
        self.device = device
        self.lags_indices = lags_indices or [0,2,4,6,8]
        self.delta_lags_indices = delta_lags_indices or [1,3,5,7,9]
        weights_lags_list = weights_lags or [1.,1.,1.,1.,1.]
        weights_delta_lags_list = weights_delta_lags or [1.,1.,1.,1.,1.]
        if decaying_weights:
            # expontential decay the is 0.1 after 10 days (assuming idx=9 is 72 hours). deltas' weights will be 1/2 of the lags'
            # i.e.: exp(t=10) = 0.1 = e^(-R*t) => R = -log(0.1)/10
            r = -np.log(0.1)/10
            weights_lags_list = [1., np.exp(-r*1), np.exp(-r*1), np.exp(-r*2), np.exp(-r*3)]
            weights_delta_lags_list = [weights_lags_list[i]/2 for i in range(len(weights_lags_list))]
        self.weights_lags = torch.tensor(weights_lags_list, device=self.device).double()  
        self.weights_delta_lags = torch.tensor(weights_delta_lags_list, device=self.device).double()  
