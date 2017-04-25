"""
To be implemented for gpu, now it uses numpy because of SVM support.
"""
import numpy as np
from keras.utils.generic_utils import get_from_module
from keras.backend.common import floatx
from keras import backend as K
from ensemble.preprocessing import *
import tensorflow as tf





@time_it_millis
@clip_extremes_tf
def avg_fusion(activations):
    return K.mean(activations, axis=0)

@clip_extremes_tf
def jrer_fusion(activations):
    return tf.multiply(jr_fusion(activations),er_fusion(activations))

@clip_extremes_tf
def jr_fusion(activations):
    jr = tf.divide(activations, tf.add(1.,tf.multiply(-1.,activations)))
    jr = tf.reduce_prod(jr, axis=0)
    return jr

@clip_extremes_tf
def er_fusion(activations):
    _max = tf.reduce_max(activations, axis=0)
    _min = tf.reduce_min(activations, axis=0)
    er = tf.divide(_max, 1-_min)
    return er

# aliases
avg = AVG = avg_fusion
jrer = JRER = jrer_fusion

def get(identifier):
    return get_from_module(identifier, globals(), 'fusions')