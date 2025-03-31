import unittest
import numpy as np
from tensor import Tensor
from optimizers import SGD


class TestSGD(unittest.TestCase):

    def test_tensor_parameter_update(self):
        param = Tensor([5.0])
        grad = np.array([2.0])
        optimizer = SGD(params=[(param, grad)], lr=0.1)
        optimizer.step()
        self.assertTrue(np.allclose(param.data, [4.8]))

    def test_tensor_gradient_handling(self):
        param = Tensor([[1.0, 2.0], [3.0, 4.0]])
        grad = Tensor([[0.1, 0.2], [0.3, 0.4]])
        optimizer = SGD(params=[(param, grad)], lr=0.5)
        optimizer.step()
        expected = np.array([[0.95, 1.9], [2.85, 3.8]])
        self.assertTrue(np.allclose(param.data, expected))

    def test_numpy_parameter_handling(self):
        param = np.array([5.0])
        original_value = param.copy()
        grad = np.array([2.0])
        optimizer = SGD(params=[(param, grad)], lr=0.1)
        optimizer.step()
        self.assertTrue(np.array_equal(param, original_value))

    def test_multiple_parameters_update(self):
        params = [
            (Tensor([1.0]), np.array([0.5])),
            (Tensor([2.0]), np.array([1.5]))
        ]
        optimizer = SGD(params=params, lr=0.2)
        optimizer.step()
        self.assertTrue(np.allclose(params[0][0].data, [0.9]))
        self.assertTrue(np.allclose(params[1][0].data, [1.7]))

    def test_high_dimensional_update(self):
        param = Tensor(np.ones((2, 3, 4)))
        grad = np.full((2, 3, 4), 0.1)
        optimizer = SGD(params=[(param, grad)], lr=0.1)
        optimizer.step()
        expected = np.full((2, 3, 4), 0.99)
        self.assertTrue(np.allclose(param.data, expected))