
import keras.backend as K
from keras.initializations import get_fans
import numpy as np


def glorot_uniform_positive(shape, name=None, dim_ordering='th'):
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(6. / (fan_in + fan_out))
    return K.random_uniform_variable(shape, 0.001, 2*s, name=name)


