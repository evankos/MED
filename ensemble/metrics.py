import numpy as np
from ensemble.common import eps
from sklearn.metrics import *

def binary_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return np.mean(np.equal(y_true, np.round(y_pred)))

def categorical_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for
    multiclass classification problems.
    '''
    # return np.random.uniform(0.3,0.4,1)
    return np.mean(np.equal(np.argmax(y_true, axis=-1),
                  np.argmax(y_pred, axis=-1)))

def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + eps())
    return precision

def map(actual, predicted):
    zip_ = np.dstack((np.round(np.clip(actual, 0, 1)),
                      predicted))
    return np.mean([ap(zip_[i,:,0],zip_[i,:,1]) for i in range(zip_.shape[0])])

def ap(y_true,y_pred):

    targets_top_k = np.where(y_true==1.)[0]
    k = targets_top_k.shape[0]
    predictions_top_k = np.argsort(y_pred)[-k:][::-1]
    states = np.in1d(predictions_top_k, targets_top_k)
    num_hits=0.
    score=0.
    for p in range(predictions_top_k.shape[0]):
        if states[p]:
            num_hits += 1.0
            score += num_hits / (p+1.0)

    return score / predictions_top_k.shape[0]


