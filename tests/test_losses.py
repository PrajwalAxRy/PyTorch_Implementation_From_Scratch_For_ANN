import numpy as np
from tensor import Tensor
from losses import MSELoss
import unittest

class TestMSELoss(unittest.TestCase):
    def test_forward_batchsize_one(self):
        mse = MSELoss()
        pred = Tensor([0.5, 2.0])
        target = Tensor([1.0, 3.0])
        loss_calculated = mse.forward(pred, target)
        loss_actual = Tensor([0.625])
        self.assertTrue(np.allclose(loss_calculated, loss_actual.data, rtol=1e-1))

    def test_forward_batchsize_multiple(self):
        mse = MSELoss()
        pred = Tensor([[0.5, 2.0], [1.5, 3.0]])
        target = Tensor([[1.0, 3.0], [1.5, 4.0]])
        loss_calculated = mse.forward(pred, target)
        loss_actual = Tensor([0.5625])
        self.assertTrue(np.allclose(loss_calculated, loss_actual.data, rtol=1e-5))

    def test_forward_oneHotConversion(self):
        mse = MSELoss()
        pred = Tensor([[0.5, 2.0, 1.0], [1.0, 0.5, 1.0]])
        target = Tensor([1, 2])
        loss_calculated = mse.forward(pred, target)
        loss_actual = Tensor([0.583])
        self.assertTrue(np.allclose(loss_calculated, loss_actual.data, rtol=1e-1))

    def test_backward_batchSize_one(self):
        mse = MSELoss()
        pred = Tensor([0.2, 0.3, 1])
        target = Tensor([0, 0, 1])
        mse.forward(pred, target)
        compute_grad = mse.backward()

        one_hot_encode = np.array([0, 0, 1])
        expected_grad = 2 * (pred.data - one_hot_encode) / pred.data.size
        self.assertTrue(np.allclose(compute_grad, expected_grad))

    def test_backward_batchSize_multiple(self):
        mse = MSELoss()
        pred = Tensor([[0.5, 2.0, 1.0], [1.0, 0.5, 1.0]])
        target = Tensor([2, 1])
        
        mse.forward(pred, target)
        compute_grad = mse.backward()

        one_hot_encode = np.array([[0, 0, 1], [0, 1, 0]])
        expected_grad = 2 * (pred.data - one_hot_encode) / pred.data.size
        self.assertTrue(np.allclose(compute_grad, expected_grad))