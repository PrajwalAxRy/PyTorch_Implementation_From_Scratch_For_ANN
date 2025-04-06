# PyTorch Replica - Artificial Neural Network (ANN) Framework from Scratch

So, here's the deal. I’ve always been curious about how deep learning libraries like PyTorch and TensorFlow work under the hood.

Instead of just using them like everyone else, I decided to roll up my sleeves and build a neural network framework from scratch – no fancy black boxes, just pure code. The goal? To understand the fundamentals of how an Artificial Neural Network (ANN) is built, layer by layer, from the ground up. 
Further, I was wanted to make a python package and not just pure code, similar to pytorch that can be called by anyone anywhere as a part of their python package.

### Why This Project?
The motivation behind the project was quite straightforward, learning and curiosity. Sure, I have used frameworks like Pytorch a lot and they are quite simple and efficient, but to actually become better at building models, I felt it was important to understand what's happening under the hood at a greater details.

### What’s Inside?
This project is basically a package that you can use just like PyTorch (well, in a simplified form, of course). It’s modular, meaning that each part of the neural network is separated into its own file, making it super easy to import and use in any order and also has test files to test the functionality of each module. 
It has the core building blocks of a neural network like:
- **Tensors**: The base data structure, similar to PyTorch’s `torch.Tensor`, but built from scratch using NumPy. This handles all the basic operations like addition, multiplication, and transpose.

- **Activation Functions**: ReLU, Sigmoid, TanH, Softmax. These are used to introduce non-linearity into the network, allowing it to learn from data. Each activation function is a class, making them easy to swap in and out as needed. I also include back propagation functionality in them. (Note: back propagation for SoftMax is quite complex)

- **Loss Functions**: Implemented Mean Squared Error (MSE) and Cross-Entropy Loss (later hehe). Whether you’re working on a regression problem or a classification task, this will handle calculating the difference between your model’s predictions and the actual values.

- **Optimizer**: I started with good old Stochastic Gradient Descent (SGD). This is what adjusts the weights of your network after each iteration, so it can minimize the error and get better over time.

- **Layers**: Fully connected layers (Dense layers) that take input, process it, and send it to the next layer. This is how the network learns and adapts based on the data it gets.

- **Model Class**: What holds everything together. You can create a model, add layers, and start training with just a few lines of code.

- **Training Loop**: Like PyTorch, I implemented a training loop that takes care of the forward pass, backward pass (backpropagation), and weight updates with the optimizer. This is where the magic of learning happens.

- **Evaluation**: To make sure your network is performing well, I’ve added a way to calculate accuracy and loss after training, so you can see how well your model is doing.

- **Data Splitting**: A small but important utility that splits your data into training and test sets, just like `train_test_split` in scikit-learn. It handles data in a way that’s familiar, so you don’t have to deal with crazy complexities.

### How to Use It?
Just like PyTorch, all you need to do is import the files and start building your neural network. You can load your data (whether it's from CSV files using Pandas or datasets like MNIST using TensorFlow), define your layers, choose your activation function, and then start training. It’s modular and simple.

Here's a quick example
```python
from tensor import Tensor
from layers import Dense
from activations import ReLU, Softmax
from losses import CrossEntropyLoss
from optimizers import SGD
from model import Model
from training import train
from evaluation import accuracy
from dataset import load_dataset

# Load your data (e.g., MNIST)
X_train, y_train, X_test, y_test = load_dataset("MNIST")

# Build the model
model = Model()
model.add(Dense(784, 128))  # Input layer to hidden layer
model.add(ReLU())  # ReLU activation
model.add(Dense(128, 10))  # Hidden layer to output layer
model.add(Softmax())  # Softmax activation for classification

# Define loss function and optimizer
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Train the model
train(model, X_train, y_train, loss_fn, optimizer, epochs=10, batch_size=64, X_val=X_test, y_val=y_test, metric_fn=accuracy)

# Evaluate the model
test_preds = model.forward(X_test)
test_loss = loss_fn.forward(test_preds, y_test)
test_acc = accuracy(test_preds, y_test)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
```

### Why should you care?
Maybe you actually done, It's fine.
At the end of the day, the project was all about building something simple yet complex (yes, implementing back prop wasn't easy), understanding nuances of neural networks.  

Feel free to fork, modify or improve it.