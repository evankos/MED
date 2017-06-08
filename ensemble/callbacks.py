from keras.callbacks import Callback
import warnings
from .metrics import *
import warnings

from keras.callbacks import Callback
import keras.backend as K
from .metrics import *


class Checkpoint(Callback):
    '''Test and save after each epoch
        validation_data: (x,y) tuple to validate on
    '''
    def __init__(self, validation_data, previous_best=0, monitor='val_loss', verbose=0,
                 save_best_only=True,
                 mode='auto', epochs_to_stop=15, obj=None):
        super(Checkpoint, self).__init__()
        self.obj = obj
        self.monitor = monitor
        self.validation_data = validation_data
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.epochs_to_stop=epochs_to_stop
        self.non_improved=0
        self.history=[]


        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'
        self.best = previous_best
        if mode == 'min':
            self.monitor_op = np.less

        elif mode == 'max':
            self.monitor_op = np.greater

        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater

            else:
                self.monitor_op = np.less

    def on_train_end(self, logs={}):
        if hasattr(self, 'best_weights'):
            self.model.set_weights(self.best_weights)

    def on_epoch_end(self, epoch, logs={}):
        feats = ['cnn', 'mfcc', 'mbh', 'sift', 'hog', 'traj']
        k = K.get_session().run(self.obj.psi)
        print("\n",k)
        k -= np.multiply(np.eye(6), k)
        maxes = np.argmax(k, axis=1)
        for index, feat in enumerate(feats):
            print(feat, feats[maxes[index]])

        if self.save_best_only:
            y_p = self.model.predict(self.validation_data[0],batch_size=128)
            current=map(self.validation_data[1], y_p)
            self.history.append(current)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    self.non_improved=0
                    if self.verbose > 0:
                        print(' : %s improved from %0.5f to %0.5f,'
                              ' saving model'
                              % ("mAP", self.best,
                                 current))
                    self.best = current
                    self.best_weights = self.model.get_weights()
                    self.best_prediction = y_p
                else:
                    self.non_improved+=1
                    if self.non_improved>self.epochs_to_stop:
                        self.model.stop_training = True
                    if self.verbose > 0:
                        print('Epoch %05d: %f did not improve' %
                              (epoch, current))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model' % (epoch))
            self.best_weights = self.model.get_weights()



class MultiCheckpoint(Callback):
    '''Test and save after each epoch
        validation_data: (x,y) tuple to validate on
    '''
    def __init__(self, validation_data, previous_best=0, monitor='val_loss', verbose=0,
                 save_best_only=True,
                 mode='auto',epochs_to_stop=15):
        super(MultiCheckpoint, self).__init__()
        self.monitor = monitor
        self.validation_data = validation_data
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.epochs_to_stop=epochs_to_stop
        self.non_improved=0
        self.history=[]


        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'
        self.best = previous_best
        if mode == 'min':
            self.monitor_op = np.less

        elif mode == 'max':
            self.monitor_op = np.greater

        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater

            else:
                self.monitor_op = np.less


    def on_epoch_end(self, epoch, logs={}):

        if self.save_best_only:
            y_p = self.model.predict(self.validation_data[0],batch_size=128)
            main = map(self.validation_data[1][:,:239], y_p[:,:239])
            l2 = map(self.validation_data[1][:,239:250], y_p[:,239:250])
            l3 = map(self.validation_data[1][:,250:], y_p[:,250:])
            current=(main+l2+l3)/3
            self.history.append(current)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    self.non_improved=0
                    if self.verbose > 0:
                        print('acc (%0.5f,%0.5f,%0.5f) : %s improved from %0.5f to %0.5f,'
                              ' saving model'
                              % (main,l2,l3, "mAP", self.best,
                                 current))
                    self.best = current
                    self.best_weights = self.model.get_weights()
                    self.best_prediction = y_p
                else:
                    self.non_improved+=1
                    if self.non_improved>self.epochs_to_stop:
                        self.model.stop_training = True
                    if self.verbose > 0:
                        print('Epoch %05d: %f did not improve' %
                              (epoch, current))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model' % (epoch))
            self.best_weights = self.model.get_weights()

