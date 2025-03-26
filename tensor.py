import numpy as np 


"""
Tensor Implementation Module

This module provides a Tensor class that mimics basic tensor operations
similar to PyTorch or NumPy. It supports operations like addition,
subtraction, multiplication, and matrix multiplication between tensors.

The Tensor class serves as the foundational data structure for our neural
network implementation. Note that gradients are handled manually in each 
layer rather than through automatic differentiation in this class.
"""

class Tensor:
    ## init function is a consuutructor for the class and is called anytime we create an object of the class
    def __init__(self, data):
        ## If data is not a numpy array, convert it to numpy array
        self.data = np.array(data, dtype=float) if not isinstance(data, np.ndarray) else data
    

    ## When I do a "c+a" and c is not a Tensor, python will first try out c.__add__(a). However, Since C is not a Tensor,
    ## it won't have __add__ method. So, it will try out a.__radd__(c) instead.
    ## So, we need to implement __radd__ method in the Tensor class.
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)
    def __radd__(self, other):
        return self.__add__(other)
    


    def __sub__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data - other_data)
    def __rsub__(self, other):    # Subtration is not commutative, so we can't just call __sub__ method
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(other_data - self.data)

    def __mul__(self, other):
        return Tensor(self.data * other.data) if isinstance(other, Tensor) else Tensor(self.data * other)
    def __rmul__(self, other):
        return self.__mul__(other)

    
    ## The __matmul__ method allows the use of the @ operator for matrix  multiplication between two tensors.
    def __matmul__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(np.matmul(self.data, other_data))
    def __rmatmul__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(np.matmul(other_data, self.data))
    
    #With @property, you can access the transpose of a Tensor instance using tensor_instance.T instead of tensor_instance.T(). This makes the code cleaner and more intuitive, especially when dealing with mathematical operations.
    @property
    def T(self):
        return Tensor(self.data.T)
    

    
    ## Example: If you have a Tensor object and you print it, the __repr__ method will be called to determine what string to display.
    ## print(tensor) based on the below implementation will return a string with the content of self.data
    def __repr__(self):
        return f"""Tensor ( 
                    {self.data} 
                    )"""