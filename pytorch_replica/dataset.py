import numpy as np
import pandas as pd

def custom_shuffle(X,y, random_state=None):
    '''
        This function shuffles the data and returns the shuffled data.
        X: Features
        y: Labels
        random_state: Random seed for reproducibility
    '''

    indices = np.arange(len(X))

    if random_state is not None:
        np.random.seed(random_state)
    
    # Shuffle the indices
    np.random.shuffle(indices)

    # Shuffle the data
    X_shuffled = X.iloc[indices] if isinstance(X, pd.DataFrame) else X[indices]
    y_shuffled = y.iloc[indices] if isinstance(y, pd.DataFrame) else y[indices]

    return X_shuffled, y_shuffled

def data_train_test_split(X, y, test_size=0.2, random_state=None):

    X, y = custom_shuffle(X, y, random_state=random_state)

    ## Split index
    split_index = int(len(X) * (1 - test_size))

    # Split the data
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, y_train, X_test, y_test