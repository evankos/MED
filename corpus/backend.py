from keras.utils.generic_utils import get_from_module

from .common import mfcc_file,cnn_file,\
    label_file,sift_file,hof_file,hog_file,\
    mbh_file,traj_file,sift_spectogram_file, class_index_file
import numpy as np
from keras.backend.common import floatx
from .WordTree import Tree


class Dataset():
    sources = {
        'mfcc':[mfcc_file,4000],
        'cnn':[cnn_file,4096],
        'sift':[sift_file,4000],
        'sift_spectogram':[sift_spectogram_file,1000],
        'traj':[traj_file,4000],
        'mbh':[mbh_file,4000],
        'hog':[hog_file,4000],
        'hof':[hof_file,4000]
    }

    def __init__(self,extend_labels=False, multilabel=False):
        self._multilabel = multilabel
        self._extend_labels = extend_labels
        self._labels = np.loadtxt(label_file(),dtype=floatx())
        self._class_index = open(class_index_file(), 'r').read().split('\n')
        if extend_labels:
            self._word_tree = Tree()
            self._extra_labels = {"layer_1":np.zeros((self._labels.shape[0],
                                                      len(self._word_tree.extra_activations["layer_1"]))),
                                  "layer_2":np.zeros((self._labels.shape[0],
                                                      len(self._word_tree.extra_activations["layer_2"]))),
                                  "layer_3":np.zeros((self._labels.shape[0],
                                                      len(self._word_tree.extra_activations["layer_3"])))}

            for sample in range(self._labels.shape[0]):
                #Removing Multilabel 1
                if not multilabel:
                    search=np.where(self._labels[sample]==1)
                    if search[0].shape[0]>1:
                        self._labels[sample]=np.zeros((self._labels.shape[1]),dtype=floatx())
                        self._labels[sample,search[0][0]]=1.
                for o_layer in list(self._word_tree.coocurrence_indexes.keys()):
                    activations = np.where(self._labels[sample]==1.)[0]
                    groups = self._word_tree.coocurrence_indexes[o_layer][activations]
                    try:
                        self._extra_labels[o_layer][sample, groups]=1.
                    except Exception:
                        print(Exception,o_layer,activations)
                        exit()

    @property
    def class_index(self):
        return self._class_index

    @class_index.setter
    def class_index(self, class_index):
        self._class_index = class_index

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self,labels):
        self._labels=labels


    def generator(self, source="mfcc", load_window=39, samples=128, categories=239, train=0, shuffle=False, continous_generation=False):
        length=self.sources[source][1]
        file=self.sources[source][0]()
        gen=True
        while gen:
            gen=continous_generation
            with open(file,'r') as source_file:
                x_batch = np.zeros((load_window * samples, length), dtype=floatx())
                y_batch = np.zeros((load_window * samples, categories), dtype=floatx())
                if self._extend_labels:
                    y_xtra_batch={"layer_1":np.zeros((load_window * samples,
                                                      self._extra_labels["layer_1"]
                                                      .shape[1]), dtype=floatx()),
                                 "layer_2":np.zeros((load_window * samples,
                                                      self._extra_labels["layer_2"]
                                                      .shape[1]), dtype=floatx()),
                                 "layer_3":np.zeros((load_window * samples,
                                                      self._extra_labels["layer_3"]
                                                      .shape[1]), dtype=floatx())}

                inner_index=0
                for index,line in enumerate(source_file):
                    if index%2==train:
                        x_batch[inner_index,:]=np.fromstring(line,dtype=floatx(),sep="\t")
                        y_batch[inner_index,:]=self._labels[index]
                        if self._extend_labels:
                            y_xtra_batch["layer_1"][inner_index,:]= self._extra_labels["layer_1"][index]
                            y_xtra_batch["layer_2"][inner_index,:]= self._extra_labels["layer_2"][index]
                            y_xtra_batch["layer_3"][inner_index,:]= self._extra_labels["layer_3"][index]
                        inner_index+=1
                        if inner_index==load_window*samples:
                            inner_index=0
                            index_array = np.arange(load_window * samples)
                            if shuffle:np.random.shuffle(index_array)
                            x_batch[:]=x_batch[index_array]
                            y_batch[:]=y_batch[index_array]
                            if self._extend_labels:
                                y_xtra_batch["layer_1"][:]= y_xtra_batch["layer_1"][index_array]
                                y_xtra_batch["layer_2"][:]= y_xtra_batch["layer_2"][index_array]
                                y_xtra_batch["layer_3"][:]= y_xtra_batch["layer_3"][index_array]
                            for yeld in range(load_window):
                                if not self._extend_labels:
                                    yield x_batch[yeld * samples:(yeld + 1) * samples], \
                                          y_batch[yeld * samples:(yeld + 1) * samples]
                                else:
                                    yield x_batch[yeld * samples:(yeld + 1) * samples],\
                                          y_batch[yeld * samples:(yeld + 1) * samples],\
                                          y_xtra_batch["layer_1"][yeld * samples:(yeld + 1) * samples],\
                                          y_xtra_batch["layer_2"][yeld * samples:(yeld + 1) * samples],\
                                          y_xtra_batch["layer_3"][yeld * samples:(yeld + 1) * samples]
                if inner_index==0:continue
                if inner_index<load_window*samples:
                    index_array = np.arange(inner_index)
                    samples=min(inner_index,samples)
                    if shuffle:np.random.shuffle(index_array)
                    x_batch[0:index_array.shape[0]]=x_batch[index_array]
                    y_batch[0:index_array.shape[0]]=y_batch[index_array]
                    if self._extend_labels:
                        y_xtra_batch["layer_1"][0:index_array.shape[0]]= y_xtra_batch["layer_1"][index_array]
                        y_xtra_batch["layer_2"][0:index_array.shape[0]]= y_xtra_batch["layer_2"][index_array]
                        y_xtra_batch["layer_3"][0:index_array.shape[0]]= y_xtra_batch["layer_3"][index_array]
                    for yeld in range(inner_index//samples):
                        if not self._extend_labels:
                            yield x_batch[yeld * samples:(yeld + 1) * samples],\
                                  y_batch[yeld * samples:(yeld + 1) * samples]
                        else:
                            yield x_batch[yeld * samples:(yeld + 1) * samples],\
                                  y_batch[yeld * samples:(yeld + 1) * samples],\
                                  y_xtra_batch["layer_1"][yeld * samples:(yeld + 1) * samples],\
                                  y_xtra_batch["layer_2"][yeld * samples:(yeld + 1) * samples],\
                                  y_xtra_batch["layer_3"][yeld * samples:(yeld + 1) * samples]

