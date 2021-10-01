import numpy as np

class Metric:
    """
        Can work for tensors as well, as long as the shape, type, device... are the consistent
    """
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

