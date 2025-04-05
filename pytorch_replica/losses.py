import numpy as np
from tensor import Tensor

class MSELoss:
    def __init__(self):
        self.last_pred = None
        self.last_target = None

    def forward(self, pred, target):
        pred_array = pred.data if isinstance(pred, Tensor) else np.array(pred)
        target_array = target.data if isinstance(target, Tensor) else np.array(target)

        #Dimensions will help me evaluates shapes lile (1,10) as well as (4,10) where we are calculating loss on 4 input samples at once
        if target_array.ndim == 1 and pred_array.ndim > 1:
            one_hot = np.zeros_like(pred_array)
            one_hot[np.arange(len(target_array)), target_array.astype(int)] = 1.0
            target_vec = one_hot
        else:
            target_vec = target_array

        self.last_pred = pred_array
        self.last_target = target_vec
        error = pred_array - target_vec
        loss = np.mean(error ** 2)
        return loss

    def backward(self):
        pred_array = self.last_pred
        target_vec = self.last_target
        grad = 2 * (pred_array - target_vec) / pred_array.size
        return grad

#TODO: MAke a crossentropy loss also



