import logging

import sys
import types

from decorator import decorator
import numpy as np
from ensemble.common import eps
import time
import tensorflow as tf
from ensemble.objectives import _to_tensor
import logging
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

@decorator
def clip_extremes(fun_fun, *args, **kwargs):
    epsilon = eps()
    args=list(args)
    for arg in range(len(args)):
        try:
            args[arg] = np.clip(args[arg], epsilon, 1. - epsilon)
        except:logging.info("Couldnt clip type: %s"%str(type(args[arg])))
    args=tuple(args)
    return fun_fun(*args, **kwargs)

@decorator
def clip_extremes_tf(fun_fun, *args, **kwargs):
    epsilon = eps()
    args=list(args)
    for arg in range(len(args)):
        try:
            if isinstance(args[arg],list):
                args[arg] = tf.stack(args[arg])
            max_value = _to_tensor(1. - epsilon, args[arg].dtype.base_dtype)
            zero = _to_tensor(0., args[arg].dtype.base_dtype)
            args[arg] = tf.clip_by_value(args[arg], zero, max_value)
        except:logging.info("Couldnt clip type: %s"%str(type(args[arg])))
    args=tuple(args)
    return fun_fun(*args, **kwargs)

def randomizer(length, validation_split = 0.3):
    pick = np.arange(length // 2)
    np.random.shuffle(pick)
    pick = pick[:int(validation_split * length / 2)]
    return pick

@decorator
def time_it_millis(fun_fun, *args, **kwargs):
    start = int(round(time.time() * 1000))
    ret = fun_fun(*args, **kwargs)
    end = int(round(time.time() * 1000))
    # logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.info("Execution of %s took %d milliseconds"%(str(fun_fun.__name__),end-start))
    return ret