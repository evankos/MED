from ..common import hierarchy_file
import numpy as np
from keras.backend.common import floatx
import numpy as np
from keras.backend.common import floatx

from ..common import hierarchy_file


class Tree():
    def __init__(self):
        self._extra_activations={"layer_1":[],
                                 "layer_2":[],
                                 "layer_3":[]}
        self._coocurrence_index_array={"layer_1":[],
                                 "layer_2":[],
                                 "layer_3":[]}
        self._coocurrence_matrix = {"layer_1":[],
                                 "layer_2":[],
                                 "layer_3":[]}
        with open(hierarchy_file(),'r') as source_file:
            for index,line in enumerate(source_file):
                synsets = line.split('-')
                if len(synsets)<=3: synsets = synsets[:-1]+["%s_next"%synsets[-2]]
                else: synsets = synsets[:-1]
                for idx, synset in enumerate(synsets):
                    if synset not in self._extra_activations["layer_%d"%(idx+1)]:
                        self._extra_activations["layer_%d"%(idx+1)].append(synset)
                for idx, hyper in enumerate(synsets):
                    extra_index = self._extra_activations["layer_%d"%(idx+1)].index(hyper)
                    self._coocurrence_index_array["layer_%d" % (idx + 1)].append(extra_index)


        for layer in list(self._extra_activations.keys()):
            self._coocurrence_index_array[layer] = np.array(self._coocurrence_index_array[layer])
            for activation_group in range(len(self._extra_activations[layer])):
                group = np.where(self._coocurrence_index_array[layer]==activation_group)[0]
                group_vector = np.zeros((len(self._coocurrence_index_array[layer])))
                group_vector[group]=1.
                self._coocurrence_matrix[layer].append(group_vector)
            self._coocurrence_matrix[layer]=np.array(self._coocurrence_matrix[layer],dtype=floatx())


    @property
    def extra_activations(self):
        return self._extra_activations

    @extra_activations.setter
    def extra_activations(self, extras):
        self._extra_activations = extras

    @property
    def coocurrence_matrix(self):
        return self._coocurrence_matrix

    @coocurrence_matrix.setter
    def coocurrence_matrix(self, cm):
        self._coocurrence_matrix = cm

    @property
    def coocurrence_indexes(self):
        return self._coocurrence_index_array

    @coocurrence_indexes.setter
    def coocurrence_indexes(self, cm):
        self._coocurrence_index_array = cm