import numpy as np

from tensor import Tensor
from model import Model
from losses import MSELoss
from optimizers import SGD


def train(model, X_train, y_train, loss_fn, optimizer, epochs=10, batch_size=32):
    '''
        Essentially we are going to train the model on the given numbers of epochs with the mnetioned optimizer.
    '''
    n_samples = X_train.shape[0]  ##X_train will be something like (100,10). With 10 being number of features and 100 being samples.
    for epoch in range(1, epochs+1):
        ## We have to shuffle the X_train and y_train data.abs
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        ## Train with the given batch size
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            # Get the batch data
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
    
            # Forward pass
            predictions = model.forward(X_batch)
            loss = loss_fn.forward(predictions, y_batch)
            epoch_loss += loss * len(X_batch)  # Accumulate the loss for the epoch. Loss is averages in optimzers, so we are getting the total loss by multiplying by the batch size

            # Backward pass
            grad_loss = loss_fn.backward()
            model.backward(grad_loss)  # Backward pass through the model to compute gradients

            # Update weights
            optimizer.step() # This will update the weights of the model based on the gradients computed in the backward pass.
        
        ## Average loss for the epoch
        epoch_loss /= n_samples  # Average the loss over all samples in the epoch
        print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}")

        #TODO: Also add some metrics, if mentioned by the user. 