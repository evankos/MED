import warnings
from abc import ABC,abstractmethod

import pickle
from keras.models import Sequential,Model
from keras.layers import Dense,Input,merge,activations
from keras.models import model_from_json
from keras.callbacks import Callback
from sklearn.svm import SVC
from sklearn.externals import joblib
from ensemble import objectives as local_objectives
from ensemble.activations import custom_mixed_activation
from ensemble.layers import CustomActivation
from keras import objectives
from keras import metrics
from keras import optimizers
import numpy as np
from keras import initializations
from keras.layers import advanced_activations,normalization
from keras import regularizers
import keras.backend as K


def to_categorical(prediction):
    classes = np.max(prediction) + 1
    y_pred = np.zeros((prediction.shape[0], classes))
    y_pred[np.arange(prediction.shape[0]), prediction] = 1
    return y_pred


class BaseClassifier(ABC):

    @abstractmethod
    def save_model(self, filepath):
        """Save Model as JSON"""
        return

    @abstractmethod
    def save_weights(self, filepath, overwrite):
        """Save Model weights as MD5"""
        return

    @abstractmethod
    def load_model(self, filepath):
        """Load Model from JSON"""
        return

    @abstractmethod
    def fit(self, x, y,batch_size,nb_epoch):
        """Train on Single Batch"""
        return

    @abstractmethod
    def train_on_generator(self,genTrain,validation_data,epochs=20,samples=2500):
        """Train on Data Generator"""
        return

    @abstractmethod
    def build(self):
        """Build and compile what is needed"""
        return

    @abstractmethod
    def predict(self,x,batch_size=32):
        """Predict on x"""
        return

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self,name):
        self._name = name


class Svm(BaseClassifier):

    def __init__(self,name='SVM',kernel='poly'):
        self.name=name
        # self.model=OneVsRestClassifier(SVC(kernel=kernel,cache_size=1000))
        self.model=SVC(kernel=kernel,cache_size=5000,verbose=True)


    def save_model(self, filepath='%s.pkl'):
        """Save Model as JSON"""
        joblib.dump(self.model, filepath % self.name)


    def save_weights(self, filepath, overwrite):
        """Save Model weights as MD5"""
        pass


    def load_model(self, filepath):
        """Load Model from JSON"""
        pass


    def slice_train(self, x, y):
        """Train on Single Batch"""
        pass


    def fit(self, x, y,batch_size=None,nb_epoch=None,validation_split=None):
        y=np.where(y==1)[1]
        self.model.fit(x,y)
        return "NO_HISTORY"

    def predict(self,x,batch_size=None):
        prediction=self.model.predict(x)
        y_pred = to_categorical(prediction)
        return y_pred

    def train_on_generator(self,genTrain,genTest,epochs=2,samples=1000):
        """Train on Data Generator"""
        pass


    def build(self):
        """Build and compile what is needed"""
        pass

    def load(self,filename):
        self.model=joblib.load(filename)


class Dnn(BaseClassifier):

    def __init__(self,input,output,name='DNN'):
        # super(Dnn, self).__init__()

        self.name=name
        model = Sequential()
        model.add(Dense(4000, input_dim=input,activation='sigmoid',
                        W_regularizer=regularizers.l2(l=0.)))
        # model.add(advanced_activations.LeakyReLU(alpha=0.1))
        model.add(normalization.BatchNormalization())
        model.add(Dense(2000,activation='sigmoid',
                        W_regularizer=regularizers.l2(l=0.)))
        # model.add(advanced_activations.LeakyReLU(alpha=0.1))
        model.add(normalization.BatchNormalization())
        model.add(Dense(output,activation='sigmoid'))
        self.check_stop = Callback()
        self.model = model

    def build(self, loss=objectives.binary_crossentropy, optimizer=optimizers.Adam()):
        # try using different optimizers and different optimizer configs
        self.model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[metrics.categorical_accuracy])



    def fit(self, x, y,batch_size=32,nb_epoch=20, validation_split=0.,validation_data=None):
        self.history = self.model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=validation_split,validation_data=validation_data, callbacks= [self.check_stop])

    def predict(self,x,batch_size=32):
        return self.model.predict(x,batch_size=batch_size)

    def predict_on_generator(self,gen,batch_size=32):
        for i,(x,y) in enumerate(gen):
            if i==0:
                y_pred = self.predict(x,batch_size=batch_size)
                y_target = y
            else:
                y_pred = np.concatenate((y_pred,self.predict(x,batch_size=batch_size)),axis=0)
                y_target = np.concatenate((y_target,y),axis=0)
        return y_pred,y_target

    def train_on_generator(self,genTrain,validation_data,epochs=20,samples=2500):

        self.history = self.model.fit_generator(genTrain,samples_per_epoch=samples,nb_epoch=epochs,callbacks=[self.check_stop])


    def save_model(self, filename='%s.json', path=""):
        json_string = self.model.to_json()
        open(path+filename % self.name, 'w').write(json_string)

    def save_weights(self, filename='%s.h5', overwrite=True, path=""):
        self.model.save_weights(path+filename % self.name, overwrite)

    def load_model(self, filepath):
        self.model = model_from_json(open(filepath).read())



class RDnn(BaseClassifier):

    def __init__(self,input,output,name='RDNN'):
        # super(Dnn, self).__init__()
        self.name=name
        features_1 = Input(shape=(input,))

        e1 = Dense(6000, input_dim=input,activation='sigmoid',
                        W_regularizer=regularizers.l2(l=0.))(features_1)
        e1 = normalization.BatchNormalization()(e1)
        f = Dense(3500,activation='sigmoid',
                        W_regularizer=regularizers.l2(l=0.))(e1)
        f = normalization.BatchNormalization()(f)
        out = Dense(output,activation='linear',name="Synsets")(f)
        out = CustomActivation(activation='custom_mixed_activation')(out)

        model = Model(input=features_1, output=out)
        self.check_stop = Callback()
        self.model = model


    def build(self,loss=objectives.binary_crossentropy, optimizer=optimizers.Adam()):
        # try using different optimizers and different optimizer configs
        self.model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[metrics.categorical_accuracy])



    def fit(self, x, y,batch_size=32,nb_epoch=20, validation_split=0.,validation_data=None):
        self.history = self.model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=validation_split,validation_data=validation_data, callbacks= [self.check_stop])

    def predict(self,x,batch_size=32):
        return self.model.predict(x,batch_size=batch_size)

    def predict_on_generator(self,gen,batch_size=32):
        for i,(x,y) in enumerate(gen):
            if i==0:
                y_pred = self.predict(x,batch_size=batch_size)
                y_target = y
            else:
                y_pred = np.concatenate((y_pred,self.predict(x,batch_size=batch_size)),axis=0)
                y_target = np.concatenate((y_target,y),axis=0)
        return y_pred,y_target

    def train_on_generator(self,genTrain,validation_data,epochs=20,samples=2500):

        self.history = self.model.fit_generator(genTrain,samples_per_epoch=samples,nb_epoch=epochs,callbacks=[self.check_stop])


    def save_model(self, filename='%s.json', path=""):
        json_string = self.model.to_json()
        open(path+filename % self.name, 'w').write(json_string)

    def save_weights(self, filename='%s.h5', overwrite=True, path=""):
        self.model.save_weights(path+filename % self.name, overwrite)

    def load_model(self, filepath):
        self.model = model_from_json(open(filepath).read())
