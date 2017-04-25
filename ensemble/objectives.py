from ensemble.common import eps
import tensorflow as tf
from keras.objectives import *
from tensorflow.contrib import metrics
import keras.backend as K


class synset_loss():
    def __init__(self, l2_cooc_mtrx, l3_cooc_mtrx, importance = 50.):
        self.l2_cooc_mtrx = tf.constant(l2_cooc_mtrx)
        self.l3_cooc_mtrx = tf.constant(l3_cooc_mtrx)
        self.importance = tf.constant(importance,dtype=tf.float32)

    def loss(self, y_true, y_pred):
        start_l2 = self.l2_cooc_mtrx.get_shape()[1]
        width_l2 = self.l2_cooc_mtrx.get_shape()[0]
        # width_l3 = self.l2_cooc_mtrx.get_shape()[0]
        aggregation_l2=tf.matmul(y_pred[:,start_l2:start_l2+width_l2],self.l2_cooc_mtrx)
        aggregation_l3=tf.matmul(y_pred[:,start_l2+width_l2:],self.l3_cooc_mtrx)
        bayes_prediction=tf.multiply(y_pred[:,:start_l2],aggregation_l2)
        bayes_prediction=tf.multiply(bayes_prediction,aggregation_l3)
        bayes_loss = binary_crossentropy(y_true[:,:start_l2],bayes_prediction)
        bayes_loss = tf.multiply(self.importance,bayes_loss)
        # target_loss = binary_crossentropy(y_true,y_pred)
        target_loss1 = binary_crossentropy(y_true[:,:start_l2],y_pred[:,:start_l2])
        target_loss2 = categorical_crossentropy(y_true[:,start_l2:start_l2+width_l2],
                                                y_pred[:,start_l2:start_l2+width_l2])
        target_loss3 = categorical_crossentropy(y_true[:,start_l2+width_l2:],
                                                y_pred[:,start_l2+width_l2:])
        total_loss = tf.add(bayes_loss,tf.add(target_loss1,tf.add(target_loss2,target_loss3)))
        return total_loss


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x
