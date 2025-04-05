import numpy as np

from tensor import Tensor

class ReLU:
    def __init__(self):
        self.last_input = None ## This will be used to calculate the gradient during backpropagation.It stores the array or tensor.
    
    def forward(self, input_data): ## Forward pass: ReLU is calculated for the whole layer at once and hence the input_data will be an array or a tensor.
        input_data_tensor = input_data.data if isinstance(input_data, Tensor) else np.array(input_data)
        self.last_input = input_data_tensor ## Storing this because we need to use this later for backpropagation.
        output = np.maximum(0, input_data_tensor)
        return Tensor(output) 
    
    def backward(self, grad_output): ## Grad_Output: represents the gradient of the loss with respect to the output of this ReLU layer.
        grad = grad_output.data if isinstance(grad_output, Tensor) else grad_output

        ## This step calulates the gradient of the loss with respect to the input of this ReLU layer.
        grad_input = grad * (self.last_input > 0).astype(float)
        return grad_input


class Sigmoid:
    def __init__(self):
        self.last_output = None
    
    def forward(self, input_data):
        input_data_tensor = input_data.data if isinstance(input_data, Tensor) else np.array(input_data)
        output = 1 / (1 + np.exp(-input_data_tensor))
        self.last_output = output
        return Tensor(output)

    def backward(self, grad_output):
        grad = grad_output.data if isinstance(grad_output, Tensor) else grad_output
        ## Detivative of sigmoid --> sigmoid_output x (1 - sigmoid_output)
        grad_input = grad * (self.last_output * (1 - self.last_output))
        return grad_input


class TanH:
    def __init__(self):
        self.last_output = None

    def forward(self, input_data):
        input_data_tensor = input_data.data if isinstance(input_data, Tensor) else np.array(input_data)
        output = (np.exp(input_data_tensor) - np.exp(-input_data_tensor)) / (np.exp(input_data_tensor) + np.exp(-input_data_tensor))  ## You can also use np.tanH
        self.last_output = output
        return Tensor(output)

    def backward(self, grad_output):
        grad = grad_output.data if isinstance(grad_output, Tensor) else grad_output
        ## Derivative: 1 - tanH_Output^2
        grad_input = grad * (1 - self.last_output ** 2)
        return grad_input


class SoftMax:
    def __init__(self):
        self.last_output = None

    def forward(self, input_data):
        input_data_tensor = input_data.data if isinstance(input_data, Tensor) else np.array(input_data)
        if input_data_tensor.ndim == 1:
            input_data_max = np.max(input_data_tensor)
            input_shifted = input_data_tensor - input_data_max
            exp_sum = np.sum(np.exp(input_shifted))
            output = np.exp(input_shifted) / exp_sum
        else:
            input_data_max = np.max(input_data_tensor, axis=1, keepdims=True)
            input_shifted = input_data_tensor - input_data_max
            exp_sum = np.sum(np.exp(input_shifted), axis=1, keepdims=True)
            output = np.exp(input_shifted) / exp_sum
        
        self.last_output = output
        return Tensor(output)

    #TODO: Check this out and learn jacobian. THIS PART IS AI GENERATED. BE CAREFUL WITH IT'S IMPLEMENTATION.
    def backward(self, grad_output):
        # Calculate gradient of loss w.r.t softmax input using the Jacobian 
        grad = grad_output.data if isinstance(grad_output, Tensor) else grad_output
        y = self.last_output  # softmax output (numpy array)
        if y.ndim == 1:
            # For single sample: use formula grad_input_i = y_i * (grad_i - sum_j(y_j * grad_j))
            s = np.sum(grad * y)
            grad_input = y * (grad - s)
        else:
            # Vectorized for batch:
            s = np.sum(grad * y, axis=1, keepdims=True)  # shape (batch, 1)
            grad_input = y * (grad - s)  # elementwise
        return grad_input

