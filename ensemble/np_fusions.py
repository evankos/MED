"""
To be implemented for gpu, now it uses numpy because of SVM support.
"""
import numpy as np
from keras.utils.generic_utils import get_from_module
from keras.backend.common import floatx
from numpy.linalg import svd
from numpy.linalg import matrix_rank
from .preprocessing import *

@time_it_millis
@clip_extremes
def avg_fusion(activations):
    return np.average(activations,axis=0)

@time_it_millis
@clip_extremes
def jrer_fusion(activations):
    jr = np.divide(activations, np.add(1.,np.multiply(-1.,activations)))
    jr = np.prod(jr, axis=0)
    _max = np.amax(activations, axis=0)
    _min = np.amin(activations, axis=0)
    er = np.divide(_max, 1-_min)
    return np.multiply(jr,er)


# aliases
avg = AVG = avg_fusion
jrer = JRER = jrer_fusion

def get(identifier):
    return get_from_module(identifier, globals(), 'fusions')