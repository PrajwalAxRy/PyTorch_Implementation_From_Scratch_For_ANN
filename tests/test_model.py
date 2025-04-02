import unittest
import numpy as np
from tensor import Tensor 
from model import Model
from layers import Dense
from activations import ReLU

class TestModel(unittest.TestCase):
    def test_add_layers(self):
        model = Model()
        model.add(Dense(10, 5))
        model.add(ReLU())
        self.assertIsInstance(model.layers[0], Dense)
        self.assertIsInstance(model.layers[1], ReLU)
        # print(model.layers)
        # first_layer = model.layers[0]
        # print("\nFirst layer object:", first_layer)
        # print("\nFirst layer weights:", first_layer.W.data)
        # print("\nFirst layer bias:", first_layer.b.data)
        # print("\nFirst layer bias:", first_layer.dw)

    def test_forward_pass_input(self):
        model = Model()
        model.add(Dense(2, 5)) ## This means input of dimension 2 and output of dimension 5
        model.add(ReLU())
        
        input_data = np.array([[1, 2], [3, 4]])
        output = model.forward(input_data)
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.data.shape, (2, 5))

        input_data_2 = Tensor(input_data)
        output_2 = model.forward(input_data_2)
        self.assertIsInstance(output_2, Tensor)
        self.assertEqual(output_2.data.shape, (2, 5))
    
    def test_parameters(self):
        model = Model()
        model.add(Dense(2, 3))
        model.add(ReLU())
        model.add(Dense(3, 1))

        params = model.parameters()
        self.assertEqual(len(params), 4) # Each Dense layer contributes 2 parameters (W, b)
        self.assertEqual(params[0][0].data.shape, (2, 3))  # W1 shape
        self.assertEqual(params[0][1].data.shape, (2,3))   # dw shape
        self.assertEqual(params[1][0].data.shape, (3,))    # b1 shape
        self.assertEqual(params[1][1].data.shape, (3,))    # db shape
        
        self.assertEqual(params[2][0].data.shape, (3, 1))  # W2 shape
        self.assertEqual(params[2][1].data.shape, (3,1))    # dw shape
        self.assertEqual(params[3][0].data.shape, (1,))    # b2 shape
        self.assertEqual(params[3][1].data.shape, (1,))    # db shape

    def test_backward_pass(self):
        model = Model()
        model.add(Dense(2, 3))
        model.add(ReLU())

        ## Forward pass
        x = Tensor([[1, 2]])
        output = model.forward(x) ## This will be of shape (1,3),this applies the xW + B

        ## Backward pass
        grad_output = Tensor([[1, 2, 3]]) # Random data that acts like the gradient of the loss w.r.t. the output
        grad_input = model.backward(grad_output)

        self.assertEqual(grad_input.data.shape, (1, 2)) # The shape of the input to the model
        self.assertIsNotNone(model.layers[0].dw) # Check if the gradients are computed
        self.assertIsNotNone(model.layers[0].db) # Check if the gradients are computed