from abc import ABC,abstractmethod
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.svm import SVC
import numpy as np

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
    def batch_train(self,x,y):
        """Train on Single Batch"""
        return

    @abstractmethod
    def fit(self, x, y,batch_size,nb_epoch,validation_split):
        """Train on Single Batch"""
        return

    @abstractmethod
    def train_on_generator(self,gen,epochs=2):
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

    def __init__(self,name='SVM'):
        self.name=name
        self.model=SVC(kernel='poly')


    def save_model(self, filepath):
        """Save Model as JSON"""
        pass


    def save_weights(self, filepath, overwrite):
        """Save Model weights as MD5"""
        pass


    def load_model(self, filepath):
        """Load Model from JSON"""
        pass


    def batch_train(self,x,y):
        """Train on Single Batch"""
        pass


    def fit(self, x, y,batch_size=None,nb_epoch=None,validation_split=None):
        y=np.where(y==1)[1]
        self.model.fit(x,y)
        self.classes=np.max(y) + 1
        return "NO_HISTORY"

    def predict(self,x,batch_size=None):
        prediction=self.model.predict(x)
        y_pred=np.zeros((prediction.shape[0],self.classes))
        y_pred[np.arange(prediction.shape[0]),prediction]=1
        return y_pred

    def train_on_generator(self,gen,epochs=2):
        """Train on Data Generator"""
        pass


    def build(self):
        """Build and compile what is needed"""
        pass



class Dnn(BaseClassifier):

    def __init__(self,input,output,width=128,name='DNN'):
        # super(Dnn, self).__init__()

        self.name=name
        model = Sequential()
        model.add(Dense(width, input_dim=input,activation='relu'))
        model.add(Dense(output,activation='sigmoid'))
        self.model=model


    def build(self):
        # try using different optimizers and different optimizer configs
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


    def batch_train(self,x,y):
        return self.model.train_on_batch(x,y)

    def fit(self, x, y,batch_size=32,nb_epoch=20,validation_split=.1):
        return self.model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=validation_split)

    def predict(self,x,batch_size=32):
        return self.model.predict(x,batch_size=batch_size)

    def train_on_generator(self,gen,epochs=2):
        for epoch in range(epochs):
            x,y=gen.get()
            self.batch_train(x,y)

    def save_model(self, filepath='DNN.json'):
        json_string = self.model.to_json()
        open(filepath, 'w').write(json_string)

    def save_weights(self,filepath='DNN.h5',overwrite=True):
        self.model.save_weights(filepath,overwrite)

    def load_model(self, filepath):
        self.model = model_from_json(open(filepath).read())
