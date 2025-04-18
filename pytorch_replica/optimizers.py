import numpy as np
from tensor import Tensor

class SGD:
    def __init__(self, params, lr=0.01):
        """
        Params: This has the list of tuples (param, grad) as provided by model.parameters()
        lr: learning rate
        """
        self.params = params
        self.lr = lr

    def step(self):
        for param, grad in self.params:
            grad = grad.data if isinstance(grad, Tensor) else grad
            if isinstance(param, Tensor):
                param.data = param.data - self.lr * grad
            else:
                param = param - self.lr * grad

#TODO: Add more Optimizers
