import numpy as np
from tensor import Tensor

class Model:
    def __init__(self):
        self.layers = []
    def add(self, layer):
        self.layers.append(layer)## This can be both RelU or Dense
    
    def forward(self, x): # x can be a numpy array or a tensor that is given as input
        ## This is the forward pass of the model. It takes the input and passes it through all the layers.
        out = x if isinstance(x, Tensor) else Tensor(x)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_output):
        ## This is the backward pass of the model. It takes the gradient of the loss with respect to the output and passes it through all the layers in reverse order.
        grad = grad_output if isinstance(grad_output, Tensor) else Tensor(grad_output)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    # This function stores the parameters of the model in a list. It is used to get the parameters of the model for optimization.   
    def parameters(self):
        '''
        model.add(Dense(2, 3))
        model.add(ReLU())
        model.add(Dense(3, 1))

        This is what Params would look like
        params = [
            (W1, dw1),  # params[0]: Layer 1 weights, shape (2, 3)
            (b1, db1),  # params[1]: Layer 1 biases, shape (3,)
            (W2, dw2),  # params[2]: Layer 2 weights, shape (3, 2)
            (b2, db2)   # params[3]: Layer 2 biases, shape (2,)
        ]

        '''
        params = []
        for layer in self.layers:
            if hasattr(layer, 'W'):
                params.append((layer.W, layer.dw))
                params.append((layer.b, layer.db))
        return params