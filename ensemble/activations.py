from keras.activations import *


def custom_mixed_activation(x):
    sig = sigmoid(x[:, :239])
    soft1 = softmax(x[:, 239:250])
    soft2 = softmax(x[:, 250:])
    sum_ = K.concatenate([sig,soft1,soft2])
    return sum_

def get(identifier):
    if identifier is None:
        return linear
    return get_from_module(identifier, globals(), 'activation function')