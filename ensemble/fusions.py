"""
To be implemented for gpu, now it uses numpy because of SVM support.
"""
import numpy as np
from keras.utils.generic_utils import get_from_module

def avg_fusion(activations):
    return np.average(activations,axis=0)





# aliases
avg = AVG = avg_fusion


def get(identifier):
    return get_from_module(identifier, globals(), 'fusions')