import numpy as np
from tensor import Tensor

class Dense:
    def __init__(self, input_dim, output_dim):
        
        self.W = Tensor(np.random.randn(input_dim, output_dim) * 0.01)
        self.b = Tensor(np.zeros(output_dim))

        self.dw = np.zeros((input_dim, output_dim))
        self.db = np.zeros(output_dim)
        self.last_input = None

    def forward(self, x):
        x_array = x.data if isinstance(x, Tensor) else np.array(x)
        self.last_input = x_array ##Save for backward prop

        ##Computing fully connected layer 
        output_result = x_array.dot(self.W.data) + self.b.data ## Shape will be (1,Output_dim) or (n,output_dim)
        return Tensor(output_result)
    
    def backward(self, grad_output):
        grad = grad_output.data if isinstance(grad_output, Tensor) else grad_output
        ## Shape of grad output -> (batch_size, output_dim)
        ##Compute gradient
        #TODO: Revisit this and add more information based on the shapes and resulting shape.
        dw = self.last_input.T.dot(grad)
        db = np.sum(grad, axis=0)

        grad_input = grad.dot(self.W.data.T)

        ##Store them again
        self.dw[...] = dw
        self.db[...] = db
        return grad_input