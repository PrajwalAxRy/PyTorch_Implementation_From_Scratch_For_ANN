import numpy as np
from tensor import Tensor

def accuracy(pred, target):
    '''
    Accuracy as the fraction of correct predictions.
    '''
    pred_array = pred.data if isinstance(pred, Tensor) else np.array(pred)
    target_array = target.data if isinstance(target, Tensor) else np.array(target)

    if pred_array.ndim > 1 and pred_array.shape[1] > 1:
        '''
        This condition checks if it's a multi class classification problem.

        pred_array.ndim: [1,2,3] is ndim = 1, while [[1,2,3],[4,5,6]] is ndim = 2.
        pred_array.shape[1]: [1,2,3] is shape = 3, while [[1,2,3],[4,5,6]] is shape = 2.
        '''
        pred_labels = np.argmax(pred_array, axis=1) # argmax will give the index of the maximum value in each row. So if pred_array is [[1,2,3],[8,6,1]], then argmax will return [2,0].
    else:
        pred_labels = (pred_array >= 0.5).astype(int)  # Assuming binary classification for 1D predictions
        pred_labels = pred_labels.flatten()  # Flatten to match target shape
    
    if target_array.ndim > 1 and target_array.shape[1] > 1:
        '''
        We do a similar check as above for target_array.
        This is for the case where target_array is one-hot encoded.
        In this case, we need to convert it to class labels.
        For example, if target_array is [[0,0,1],[1,0,0]], we want to convert it to [2,0].
        '''
        target_labels = np.argmax(target_array, axis=1)
    else:
        target_labels = target_array.astype(int).flatten()  # Flatten to match pred shape

    ## Calculating accuracy
    accuracy_value = np.mean(pred_labels == target_labels)  # Fraction of correct predictions
    return accuracy_value