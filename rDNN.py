
import keras.optimizers as optimizer
import numpy as np
from keras import metrics
from keras.engine import Merge
from keras.layers import Activation, Dense
from keras.models import Sequential

from corpus.backend import Dataset
from ensemble import objectives
from ensemble.callbacks import Checkpoint
from ensemble.classifiers import Dnn
from ensemble.metrics import map
from ensemble.np_functions import jrer_fusion as jrer_fusion_np
from ensemble.preprocessing import randomizer
from ensemble.tf_functions import jrer_fusion
from ensemble.regularizers import TracenormRegularizer

import keras.backend as K
from keras.layers import normalization
from keras import regularizers
from ensemble.initializations import glorot_uniform_positive


import tensorflow as tf

def feature_dnn(input, output):
    model = Sequential()
    model.add(Dense(output, input_dim=input, activation='sigmoid',
                    W_regularizer=regularizers.l2(l=0.)))
    model.add(normalization.BatchNormalization())
    return model


dataset=Dataset(multilabel=True)
# replace 5000 with dataset.labels.shape[0] to use all the data (slow and mem intensive).
sample_number = 11000
e_width = 1000
fusion_width = 2000
net_output = 239
# random indexer to retrieve 30% of the test data for validation
pick=randomizer(sample_number//2)

# loading the train datasets.
cnn_x,\
cnn_y=next(dataset.generator(source='cnn', samples=sample_number, load_window=1, train=1))

mfcc_x,\
mfcc_y=next(dataset.generator(source='mfcc', samples=sample_number, load_window=1, train=1))

mbh_x,\
mbh_y=next(dataset.generator(source='mbh', samples=sample_number, load_window=1, train=1))

sift_x,\
sift_y=next(dataset.generator(source='sift', samples=sample_number, load_window=1, train=1))

hog_x,\
hog_y=next(dataset.generator(source='hog', samples=sample_number, load_window=1, train=1))

traj_x,\
traj_y=next(dataset.generator(source='traj', samples=sample_number, load_window=1, train=1))




# loading the test datasets.
cnn_x_t,\
cnn_y_t=next(dataset.generator(source='cnn', samples=sample_number, load_window=1, train=0))

mfcc_x_t,\
mfcc_y_t=next(dataset.generator(source='mfcc', samples=sample_number, load_window=1, train=0))

mbh_x_t,\
mbh_y_t=next(dataset.generator(source='mbh', samples=sample_number, load_window=1, train=0))

sift_x_t,\
sift_y_t=next(dataset.generator(source='sift', samples=sample_number, load_window=1, train=0))

hog_x_t,\
hog_y_t=next(dataset.generator(source='hog', samples=sample_number, load_window=1, train=0))

traj_x_t,\
traj_y_t=next(dataset.generator(source='traj', samples=sample_number, load_window=1, train=0))


regularizer = TracenormRegularizer(lr=.001, modalities=6, e_width=e_width)
# regularizer = regularizers.l2(l=0.)

cnn_E_classifier = feature_dnn(dataset.sources['cnn'][1], e_width)
mfcc_E_classifier = feature_dnn(dataset.sources['mfcc'][1], e_width)
mbh_E_classifier = feature_dnn(dataset.sources['mbh'][1], e_width)
sift_E_classifier = feature_dnn(dataset.sources['sift'][1], e_width)
hog_E_classifier = feature_dnn(dataset.sources['hog'][1], e_width)
traj_E_classifier = feature_dnn(dataset.sources['traj'][1], e_width)

model = Sequential()
model.add(Merge([
    cnn_E_classifier,
    mfcc_E_classifier,
    mbh_E_classifier,
    sift_E_classifier,
    hog_E_classifier,
    traj_E_classifier
], mode='concat'))
model.add(Dense(fusion_width,activation='sigmoid',
                W_regularizer=regularizer))
model.add(normalization.BatchNormalization())
model.add(Dense(net_output,activation='sigmoid'))

model.compile(loss=objectives.binary_crossentropy,
                      optimizer=optimizer.SGD(lr=0.2),
                      metrics=[metrics.categorical_accuracy])


#callback to save the best weights and trigger stop if needed
check_stop = Checkpoint(validation_data=(
    [cnn_x_t[pick],
     mfcc_x_t[pick],
     mbh_x_t[pick],
     sift_x_t[pick],
     hog_x_t[pick],
     traj_x_t[pick]],
    cnn_y_t[pick]),previous_best=0.,verbose=1,mode='max',epochs_to_stop=15,obj=regularizer)



#starting the training
model.fit([cnn_x, mfcc_x, mbh_x, sift_x, hog_x, traj_x], cnn_y, batch_size=256, nb_epoch=500, callbacks=[check_stop], shuffle=True)







# model.load_weights("weights/%s_output_RDNN_.h5" % 'mfcc_cnn_mbh_sift_hog_traj')

#predict and save on the test dataset
y_p = model.predict([cnn_x_t,
                     mfcc_x_t,
                     mbh_x_t,
                     sift_x_t,
                     hog_x_t,
                     traj_x_t])
np.save("outputs/%s_output_RDNN_" % 'mfcc_cnn_mbh_sift_hog_traj', y_p)
# model.save_weights("weights/%s_output_RDNN_.h5" % 'mfcc_cnn_mbh_sift_hog_traj', overwrite=True)
print("mAP", map(cnn_y_t,y_p))





