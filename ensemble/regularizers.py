from keras.regularizers import Regularizer
import keras.backend as K
from ensemble import trace_norm, matrix_inverse, symsqrt, reshape, cast, sqrt, matmul, multiply_elemwise, variable
from keras.backend.common import floatx
import numpy as np




class TracenormRegularizer(Regularizer):

    def __init__(self, lr, e_width=1000, modalities=6):
        self.lr = cast(lr,dtype='float32')
        self.modalities = cast(modalities,dtype='int32')
        self.e_width = cast(e_width,dtype='int32')

        self.uses_learning_phase = True

    def set_param(self, p):
        if hasattr(self, 'p'):
            raise Exception('Regularizers cannot be reused. '
                            'Instantiate one regularizer per layer.')
        self.p = p

    def __call__(self, loss):
        _, dim2 = K.eval(K.shape(self.p))
        self.We = K.transpose(reshape(self.p, [self.modalities, self.e_width * dim2]))
        if K.ndim(self.We) > 2:
            raise Exception('Tracenorm regularizer '
                            'is only available for dense '
                            'and embedding layers.')
        self.cauchy_schwarz = symsqrt(matmul(self.We, self.We))
        self.tr = cast(trace_norm(cast(self.cauchy_schwarz,dtype='complex64')),dtype='float32')
        self.psi = multiply_elemwise(1/self.tr,self.cauchy_schwarz)
        self.psi_inv = matrix_inverse(self.psi)
        # tr(A'B)=tr(AB') doing this to save memory
        self.correlations = matmul(matmul(self.We, self.psi_inv, transpose_a=False),self.We)
        self.delta = self.lr * cast(trace_norm(cast(self.correlations,'complex64')),'float32')
        regularized_loss = loss + self.delta
        return K.in_train_phase(regularized_loss, loss)