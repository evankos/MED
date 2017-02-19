from abc import ABC,abstractmethod
from .common import project_root
from os.path import join
from . import fusions
import numpy as np


class Ensemble(ABC):

    @property
    def lateFusion(self):
        return self._lateFusion

    @lateFusion.setter
    def lateFusion(self,fusion):
        self._lateFusion = fusion

    @property
    def histories(self):
        return self._histories

    @histories.setter
    def histories(self,history):
        try:
            self._histories
        except :
            self._histories=[history]
        else:
            self._histories.append(history)

    @abstractmethod
    def train_on_batch(self, data):
        """Implement the training of all classifiers"""
        return



class Parallel(Ensemble):


    def __init__(self, classifiers=[]):
        self.classifiers = []
        for classifier in classifiers:
            self.add(classifier)

    def add(self, classifier):
        self.classifiers.append(classifier)

    def compile(self,fusion='avg_fusion'):
        self.lateFusion=fusions.get(fusion)
        for classifier in self.classifiers:
            classifier.build()

    def train_on_batch(self, data):
        for idx,classifier in enumerate(self.classifiers):
            print(classifier.batch_train(data[idx][0],data[idx][1]))

    def fit(self, data, epochs=4):
        for idx,classifier in enumerate(self.classifiers):
            print("Training Classifier %s" % classifier.name)
            self.histories=classifier.fit(data[idx][0],data[idx][1],nb_epoch=epochs)

    def predict(self,data,batch=32):
        y_all=[]
        for idx,classifier in enumerate(self.classifiers):
            y_all.append(classifier.predict(data[idx],batch_size=32).ravel())
        print(y_all[0])
        print(y_all[1])
        return self.lateFusion(np.array(y_all))


    def save_models(self,file="model"):
        for classifier in self.classifiers:
            path=join(project_root(),file,"%s.json"%classifier.name)
            classifier.save_model(filepath=path)

    def save_weights(self,file="weights"):
        for classifier in self.classifiers:
            path=join(project_root(),file,"%s.h5"%classifier.name)
            classifier.save_weights(filepath=path,overwrite=True)