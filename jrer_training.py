

from keras.engine import Merge
from ensemble import objectives
from ensemble.callbacks import Checkpoint
from keras import metrics
from corpus.backend import Dataset
import numpy as np

from ensemble.classifiers import Dnn
from ensemble.preprocessing import randomizer
from keras.models import Model, Sequential
from keras.layers import Dense,Input,Activation
from keras import regularizers
from keras.layers import normalization
from ensemble.tf_fusions import er_fusion,jr_fusion,jrer_fusion
from ensemble.np_fusions import jr_fusion as jr_fusion_np
from ensemble.np_fusions import er_fusion as er_fusion_np
from ensemble.metrics import map
from keras.optimizers import SGD, Adam
import keras.backend as K
import tensorflow as tf


dataset=Dataset(multilabel=True)
# replace 5000 with dataset.labels.shape[0] to use all the data (slow and mem intensive).
sample_number = dataset.labels.shape[0]

# random indexer to retrieve 30% of the test data for validation
pick=randomizer(sample_number)




# loading the train datasets.
cnn_x,\
cnn_y=next(dataset.generator(source='cnn', samples=sample_number, load_window=1, train=1))

cnn_x_t,\
cnn_y_t=next(dataset.generator(source='cnn', samples=sample_number, load_window=1, train=0))


# loading the test datasets.
mfcc_x,\
mfcc_y=next(dataset.generator(source='mfcc', samples=sample_number, load_window=1, train=1))

mfcc_x_t,\
mfcc_y_t=next(dataset.generator(source='mfcc', samples=sample_number, load_window=1, train=0))






cnn_dnn=Dnn(dataset.sources['cnn'][1], 239, name='DNN_cnn')
# cnn_dnn.load_weights(path="weights/")
mfcc_dnn=Dnn(dataset.sources['mfcc'][1], 239, name='DNN_mfcc')
# mfcc_dnn.load_weights(path="weights/")

model = Sequential()
model.add(Merge([cnn_dnn.model, mfcc_dnn.model], mode=jrer_fusion, output_shape=(239,)))
model.add(Activation('sigmoid'))

model.compile(loss=objectives.binary_crossentropy,
                      optimizer=SGD(lr=0.03),
                      metrics=[metrics.categorical_accuracy])


#callback to save the best weights and trigger stop if needed
check_stop = Checkpoint(validation_data=([cnn_x_t[pick],mfcc_x_t[pick]],cnn_y_t[pick]),
                                             previous_best=0.,
                                             verbose=1,mode='max',epochs_to_stop=40)


#starting the training
model.fit([cnn_x,mfcc_x], cnn_y, batch_size=128, nb_epoch=200, callbacks=[check_stop])


if hasattr(check_stop, 'best_weights'):
    model.set_weights(check_stop.best_weights)

#predict and save on the test dataset
y_p = model.predict([cnn_x_t,mfcc_x_t])
np.save("outputs/%s_output_MEAN_INCORPORATED_DNN_"%'mfcc_cnn',y_p)















