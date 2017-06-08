"""
To be implemented for gpu, now it uses numpy because of SVM support.
"""
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import _to_tensor
from keras.utils.generic_utils import get_from_module

from ensemble.preprocessing import *


@time_it_millis
@clip_extremes_tf
def avg_fusion(activations):
    return K.mean(activations, axis=0)

@clip_extremes_tf
def prod(activations):
    return tf.reduce_prod(activations, axis=0)

def jrer_fusion(activations):
    return tf.multiply(tf.exp(jr_fusion(activations)),er_fusion(activations))

@clip_extremes_tf
def jr_fusion(activations):
    jr = tf.log(tf.divide(activations, tf.add(1.,tf.multiply(-1.,activations))))
    jr = tf.reduce_sum(jr, axis=0)
    return jr

@clip_extremes_tf
def er_fusion(activations):
    _max = tf.reduce_max(activations, axis=0)
    _min = tf.reduce_min(activations, axis=0)
    er = tf.divide(_max, 1-_min)
    return er


def trace_norm(w):
    return tf.trace(w)

def matrix_inverse(w):
    return tf.matrix_inverse(w)

def reshape(w,shape):
    return tf.reshape(w,shape)

def cast(v,dtype):
    return tf.cast(v, _convert_string_dtype(dtype))

def sqrt(w):
    return tf.sqrt(w)

def transpose(w):
    return tf.transpose(w)

def matmul(a,b,transpose_a = True, transpose_b = False):
    return tf.matmul(a,b,transpose_a=transpose_a,transpose_b=transpose_b)

def multiply_elemwise(k, w):
    return tf.multiply(k,w)

def divide(a,b):
    return tf.divide(a,b)

def clip_negatives(w,min=eps()):
    max_value = _to_tensor(3000, w.dtype.base_dtype)
    zero = _to_tensor(min, w.dtype.base_dtype)
    return tf.clip_by_value(w, zero, max_value)

def variable(w, dtype, name=None):
    return tf.Variable(w, dtype=_convert_string_dtype(dtype), name=name)

def symsqrt(mat, iters=1):
    """Symmetric square root."""
    # s, u, v = tf.svd(mat)
    # # sqrt is unstable around 0, just use 0 in such case
    # si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
    # return tf.matmul(tf.matmul(u,tf.diag(si)),tf.transpose(v))

    # Chebyshev
    # second = np.add(I,np.multiply(-1,np.matmul(A_,XX)))
    # third = np.add(I,np.multiply(-1,np.matmul(A_,XX)))
    # X = np.matmul(X,np.add(np.add(I,np.multiply(1/2,second)),
    #                    np.multiply(3/8,np.matmul(third,third))))

    _, dim = K.eval(tf.shape(mat))
    A_ = matrix_inverse(mat)
    X = tf.eye(dim)
    I = tf.eye(dim)
    # a,b = tf.constant(0.5),tf.constant(-1.0)
    for iter in range(iters):
        # Newton method
        X = tf.matmul(X, tf.add(I, tf.multiply(0.5, tf.add(I, tf.multiply(-1.0, tf.matmul(A_, tf.matmul(X, X)))))))
    return X

def _convert_string_dtype(dtype):
    if dtype == 'float16':
        return tf.float16
    elif dtype == 'float32':
        return tf.float32
    elif dtype == 'float64':
        return tf.float64
    elif dtype == 'complex64':
        return tf.complex64
    elif dtype == 'int16':
        return tf.int16
    elif dtype == 'int32':
        return tf.int32
    elif dtype == 'int64':
        return tf.int64
    elif dtype == 'uint8':
        return tf.int8
    elif dtype == 'uint16':
        return tf.uint16
    else:
        raise ValueError('Unsupported dtype:', dtype)

# aliases
avg = AVG = avg_fusion
jrer = JRER = jrer_fusion

def get(identifier):
    return get_from_module(identifier, globals(), 'fusions')