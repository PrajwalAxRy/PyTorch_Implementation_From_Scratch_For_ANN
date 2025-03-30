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
        loss_actual = Tensor([0.25, 1.0])
        self.assertAlmostEqual(loss_calculated, loss_actual.data, places=5)

    def test_forward_batchsize_multiple(self):
        mse = MSELoss()
        pred = Tensor([[0.5, 2.0], [1.5, 3.0]])
        target = Tensor([[1.0, 3.0], [2.0, 4.0]])
        loss_calculated = mse.forward(pred, target)
        loss_actual = Tensor([0.25, 1.0])
        self.assertAlmostEqual(loss_calculated, loss_actual.data, places=5)

    def test_forward_oneHotConversion(Self):
        mse = MSELoss()
        pred = Tensor([[0.5, 2.0, 3.0], [4.0, 1.5, 3.0]])
        target = Tensor([1, 0])
        loss_calculated = mse.forward(pred, target)
        loss_actual = Tensor([0.25, 1.0])
        self.assertAlmostEqual(loss_calculated, loss_actual.data, places=5)

    def test_backward_batchSize_one(self):
        mse = MSELoss()
        pred = Tensor([0.2, 0.3, 0.5])
        target = Tensor([2])

        mse_loss.forward(pred, target)
        compute_grad = mse_loss.backward()

        one_hot_encode = np.array([0, 0, 1])
        expected_grad = 2 * (pred.data - one_hot_encode) / pred.size
        self.assertTrue(np.allclose(compute_grad, expected_grad))

    def test_backward_batchSize_multiple(self):
        mse = MSELoss()
        pred = Tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.6]])
        target = Tensor([[2], [1]])
        
        mse_loss.forward(pred, target)
        compute_grad = mse_loss.backward()

        one_hot_encode = np.array([[0, 0, 1], [0, 1, 0]])
        expected_grad = 2 * (pred.data - one_hot_encode) / pred.size
        self.assertTrue(np.allclose(compute_grad, expected_grad))