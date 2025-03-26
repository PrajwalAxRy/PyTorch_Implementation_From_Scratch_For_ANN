import unittest
import numpy as np

from pytorch_implementation_from_scratch_for_ann.tensor import Tensor

class TestTensor(unittest.TestCase):
    def test_initialization(self):
        t1 = Tensor([1, 2, 3])
        self.assertTrue(np.array_equal(t1.data, np.array([1,2,3])))

        t2 = Tensor(np.array([5,6,7]))
        self.assertTrue(np.array_equal(t2.data, np.array([5,6,7])))
    
    def test_addition(self):
        t1 = Tensor([1,2,3])
        t2 = Tensor([1,2,3])
        result = t1 + t2
        self.assertTrue(np.array_equal(result.data, np.array([2,4,6])))

        result = t1 + 1
        self.assertTrue(np.array_equal(result.data, np.array([2,3,4])))

        result = 1 + t1
        self.assertTrue(np.array_equal(result.data, np.array([2,3,4])))

    def test_subtraction(self):
        t1 = Tensor([4, 5, 6])
        t2 = Tensor([1, 2, 3])
        result = t1 - t2
        self.assertTrue(np.array_equal(result.data, np.array([3, 3, 3])))

        result = t1 - 1
        self.assertTrue(np.array_equal(result.data, np.array([3, 4, 5])))

        result = 10 - t1
        self.assertTrue(np.array_equal(result.data, np.array([6, 5, 4])))

    def test_multiplication(self):
        t1 = Tensor([1,2,3])
        t2 = Tensor([2,3,4])
        result = t1 * t2
        self.assertTrue(np.array_equal(result.data, np.array([2, 6, 12])))
    
        result = t1 * 2
        self.assertTrue(np.array_equal(result.data, np.array([2, 4, 6])))

        result = 2 * t1
        self.assertTrue(np.array_equal(result.data, np.array([2, 4, 6])))

    def test_matrix_multiplication(self):
        t1 = Tensor([[1, 2], [3, 4]])
        t2 = Tensor([[5, 6], [7, 8]])
        result = t1 @ t2

        self.assertTrue(np.array_equal(result.data, np.array([[19, 22], [43, 50]])))
        
        result = t1 @ np.array([[5, 6], [7, 8]])
        self.assertTrue(np.array_equal(result.data, np.array([[19, 22], [43, 50]])))
    
    def test_transpose(self):
        t1 = Tensor([[1, 2], [3, 4]])
        
        self.assertTrue(np.array_equal(t1.T.data, np.array([[1, 3], [2, 4]])))

    def repr_test(self):
        t1 = Tensor([1, 2, 3])

        # Test 1: Direct representation in console
        t1

        # Test 2: Using print
        print(t1)

        # Test 3: Using repr() function
        print(repr(t1))

        # Test 4: In a collection
        tensors = [Tensor([1, 2]), Tensor([3, 4])]
        print(tensors)
