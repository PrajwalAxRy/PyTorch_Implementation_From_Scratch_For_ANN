import numpy as np 
import pandas as pd

from tensorflow.keras.datasets import mnist

from layers import Dense 
from activations import ReLU, Softmax 
from losses import MSELoss
from optimizers import SGD
from model import Model
from training import train
from evaluation import accuracy
from dataset import data_train_test_split



dataset_name = 'mnist'
loss_metric = 'mse' ##Useless

(X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()
X_train_full = X_train_full.reshape(-1, 28*28).astype('float32') / 255.0
X_test_full = X_test_full.reshape(-1, 28*28).astype('float32') / 255.0

X_train, y_train, X_test, y_test = data_train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

input_size = X_train.shape[1]
num_classes = len(np.unique(y_train))

model = Model()
model.add(Dense(input_size, 128))
model.add(ReLU())
model.add(Dense(128, num_classes))
model.add(Softmax())

loss_fn = MSELoss()