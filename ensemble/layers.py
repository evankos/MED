from keras.backend import floatx
import numpy as np
from ensemble import activations
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf


class ScalarLayer(Layer):

    def __init__(self, **kwargs):
        super(ScalarLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scalar = K.variable(1.0,dtype=floatx())
        self.trainable_weights = [self.scalar]
        self.regularizers = []
        self.constraints = {}
        super(ScalarLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        return x*self.scalar#K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape


class VectorLayer(Layer):

    def __init__(self, **kwargs):
        super(VectorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.vector = K.ones((239,), dtype=floatx())
        self.trainable_weights = [self.vector]
        self.regularizers = []
        self.constraints = {}
        super(VectorLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        return tf.multiply(x,self.vector)#x*self.vector#K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape


class CustomActivation(Layer):
    '''Applies an activation function to an output.

    # Arguments
        activation: name of activation function to use
            (see: [activations](../activations.md)),
            or alternatively, a Theano or TensorFlow operation.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    '''
    def __init__(self, activation, **kwargs):
        self.supports_masking = True
        self.activation = activations.get(activation)
        super(CustomActivation, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return self.activation(x)

    def get_config(self):
        config = {'activation': self.activation.__name__}
        base_config = super(CustomActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

