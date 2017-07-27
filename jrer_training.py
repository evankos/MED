

import keras.optimizers as optimizer
import numpy as np
from keras import metrics
from keras.engine import Merge, InputLayer
from keras.layers import Activation,Input
from keras.models import Sequential

from corpus.backend import Dataset
from ensemble import objectives
from ensemble.callbacks import SmartCheckpoint
from ensemble.classifiers import Dnn
from ensemble.layers import ScalarLayer, VectorLayer
from ensemble.metrics import map
from ensemble.np_functions import jrer_fusion as jrer_fusion_np
from ensemble.np_functions import avg_fusion as avg_fusion_np
from ensemble.preprocessing import randomizer
from ensemble.tf_functions import jrer_fusion,avg_fusion
from ensemble.optimizers import YFOptimizer
from keras.optimizers import TFOptimizer


def freeze(model):
    for layer in model.layers:
        layer.trainable = False

# define your optimizer
opt = TFOptimizer(YFOptimizer())

dataset=Dataset(multilabel=True)
# replace 5000 with dataset.labels.shape[0] to use all the data (slow and mem intensive).
sample_number = dataset.labels.shape[0]


# random indexer to retrieve 30% of the test data for validation
pick=randomizer(sample_number//2,validation_split=1.0)




# loading the train datasets.
_,\
y=next(dataset.generator(source='cnn', samples=sample_number, load_window=1, train=1))

cnn_x,\
_=np.load("inputs/%s_output_dnn_.npy"%'cnn'),y

mfcc_x,\
_=np.load("inputs/%s_output_dnn_.npy"%'mfcc'),y

mbh_x,\
_=np.load("inputs/%s_output_dnn_.npy"%'mbh'),y

hog_x,\
_=np.load("inputs/%s_output_dnn_.npy"%'hog'),y

traj_x,\
_=np.load("inputs/%s_output_dnn_.npy"%'traj'),y


# loading the test datasets.
_,\
yt=next(dataset.generator(source='cnn', samples=sample_number, load_window=1, train=0))

cnn_x_t,\
_=np.load("outputs/%s_output_dnn_.npy"%'cnn'),yt

mfcc_x_t,\
_=np.load("outputs/%s_output_dnn_.npy"%'mfcc'),yt

mbh_x_t,\
_=np.load("outputs/%s_output_dnn_.npy"%'mbh'),yt

hog_x_t,\
_=np.load("outputs/%s_output_dnn_.npy"%'hog'),yt

traj_x_t,\
_=np.load("outputs/%s_output_dnn_.npy"%'traj'),yt

# cnn_dnn=Dnn(dataset.sources['cnn'][1], 239, name='DNN_cnn')
# cnn_dnn.load_weights(path="weights/")
# freeze(cnn_dnn.model)
#
# mfcc_dnn=Dnn(dataset.sources['mfcc'][1], 239, name='DNN_mfcc')
# mfcc_dnn.load_weights(path="weights/")
# freeze(mfcc_dnn.model)
#
# mbh_dnn=Dnn(dataset.sources['mbh'][1], 239, name='DNN_mbh')
# mbh_dnn.load_weights(path="weights/")
# freeze(mbh_dnn.model)
#
# hog_dnn=Dnn(dataset.sources['hog'][1], 239, name='DNN_hog')
# hog_dnn.load_weights(path="weights/")
# freeze(hog_dnn.model)
#
# traj_dnn=Dnn(dataset.sources['traj'][1], 239, name='DNN_traj')
# traj_dnn.load_weights(path="weights/")
# freeze(traj_dnn.model)

cnn_dnn = Sequential()
cnn_dnn.add(InputLayer(input_shape=(239,)))
cnn_scalar=VectorLayer()
cnn_dnn.add(cnn_scalar)
# cnn_dnn.add(Activation('sigmoid'))

mfcc_dnn = Sequential()
mfcc_dnn.add(InputLayer(input_shape=(239,)))
mfcc_scalar=VectorLayer()
mfcc_dnn.add(mfcc_scalar)
# mfcc_dnn.add(Activation('sigmoid'))

mbh_dnn = Sequential()
mbh_dnn.add(InputLayer(input_shape=(239,)))
mbh_scalar=VectorLayer()
mbh_dnn.add(mbh_scalar)
# mbh_dnn.add(Activation('sigmoid'))

hog_dnn = Sequential()
hog_dnn.add(InputLayer(input_shape=(239,)))
hog_scalar=VectorLayer()
hog_dnn.add(hog_scalar)
# hog_dnn.add(Activation('sigmoid'))

traj_dnn = Sequential()
traj_dnn.add(InputLayer(input_shape=(239,)))
traj_scalar=VectorLayer()
traj_dnn.add(traj_scalar)
# traj_dnn.add(Activation('sigmoid'))




model = Sequential()
model.add(Merge([cnn_dnn, mfcc_dnn, mbh_dnn, hog_dnn, traj_dnn], mode=jrer_fusion, output_shape=(239,)))
model.add(Activation('linear'))

model.compile(loss=objectives.binary_crossentropy,
                      optimizer=optimizer.Adam(),
                      metrics=[metrics.categorical_accuracy])

model.summary()

# model.load_weights(filepath='weights/JRER_TRAINING.h5')

yp1=cnn_x_t
yp2=mfcc_x_t
yp3=mbh_x_t
yp4=hog_x_t
yp5=traj_x_t

print("cnn",map(yt,yp1))
print("mfcc",map(yt,yp2))
print("mbh",map(yt,yp3))
print("hog",map(yt,yp4))
print("traj",map(yt,yp5))
print("jrer",map(yt,jrer_fusion_np(np.array([yp1,yp2,yp3,yp4,yp5]))))
print("avg", map(yt, avg_fusion_np(np.array([yp1, yp2, yp3, yp4, yp5]))))


y_p = model.predict([cnn_x_t,mfcc_x_t,mbh_x_t,hog_x_t,traj_x_t])
prev_best=map(yt,y_p)
print("before", prev_best)




#callback to save the best weights and trigger stop if needed
check_stop = SmartCheckpoint(validation_data=([cnn_x_t[pick], mfcc_x_t[pick], mbh_x_t[pick], hog_x_t[pick], traj_x_t[pick]], yt[pick]),
                             previous_best=prev_best,
                             verbose=1, mode='max', epochs_to_stop=100,
                             extra_obj_list=[cnn_scalar.vector,mfcc_scalar.vector,
                                             mbh_scalar.vector,hog_scalar.vector,
                                             traj_scalar.vector])


#starting the training
model.fit([cnn_x, mfcc_x, mbh_x, hog_x, traj_x], y, batch_size=256, nb_epoch=200, callbacks=[check_stop],verbose=2)















