from .common import project_root
from os.path import join

class Ensemble(object):

    @property
    def lateFusion(self):
        return self._lateFusion

    @lateFusion.setter
    def lateFusion(self,fusion):
        self._lateFusion = fusion



class Parallel(Ensemble):

    def __init__(self, classifiers=[], fusion='avg'):
        self.lateFusion=fusion

        self.classifiers = []
        for classifier in classifiers:
            self.add(classifier)

    def add(self, classifier):
        self.classifiers.append(classifier)

    def save_models(self,file="model"):
        for classifier in self.classifiers:
            path=join(project_root(),file,"%s.json"%classifier.name)
            classifier.save_model(filepath=path)

    def save_weights(self,file="weights"):
        for classifier in self.classifiers:
            path=join(project_root(),file,"%s.h5"%classifier.name)
            classifier.save_weights(filepath=path,overwrite=True)