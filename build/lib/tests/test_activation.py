import unittest
import numpy as np

from tensor import Tensor
from activations import ReLU, Sigmoid, TanH, SoftMax

class TestActivation(unittest.TestCase):

    def test_ReLU_forward(self):
        relu = ReLU()
        x = Tensor([1,-2,3,-4,5])
        output = relu.forward(x)
        self.assertTrue(np.array_equal(output.data, np.array([1,0,3,0,5])))

    def test_ReLU_backward(self):
        relu = ReLU()
        x = Tensor([1,-2,3,-4,5])
        relu.forward(x) ## This helps to set the input for the backward pass
        grad_input = relu.backward(Tensor([1,2,3,4,5]))
        self.assertTrue(np.array_equal(grad_input.data, np.array([1,0,3,0,5])))

    
    def test_Sigmoid_forward(self):
        sigmoid = Sigmoid()
        x = Tensor([0, 1, -1])
        output = sigmoid.forward(x)
        expected_output = 1 / (1 + np.exp(-x.data))
        np.testing.assert_allclose(output.data, expected_output, atol=1e-5) # atol is absolute tolerance.
    
    #NOTE: Had to fix issues with "memoryview" and "ndarray" in the below test case.
    def test_Sigmoid_backward(self):
        sigmoid = Sigmoid()
        x = Tensor([0, 1, -1])
        sigmoid.forward(x)
        grad_output = Tensor([0.1, 0.2, 0.3])
        grad_input = sigmoid.backward(grad_output)
        
        # Calculate expected gradient using NumPy operations
        last_output_array = np.array(sigmoid.last_output.data)
        grad_output_array = np.array(grad_output.data)
        expected_grad_array = grad_output_array * (last_output_array * (1 - last_output_array))
        
        np.testing.assert_allclose(grad_input.data, expected_grad_array, atol=1e-5) # atol is absolute tolerance.
    
    def test_tanh_forward(self):
        tanh = TanH()
        x = Tensor([0.0])
        out = tanh.forward(x)
        self.assertAlmostEqual(out.data[0], 0.0, places=5)

    def test_tanh_backward(self):
        tanh = TanH()
        x = Tensor([0.0])
        tanh.forward(x)
        grad_input = tanh.backward(Tensor([1.5]))
        self.assertAlmostEqual(grad_input.data[0], 1.5, places=5)


if __name__ == '__main__':
    unittest.main()