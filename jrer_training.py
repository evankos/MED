from keras.engine import Merge

from ensemble import objectives
from ensemble.callbacks import Checkpoint
from keras import metrics
from corpus.backend import Dataset
import numpy as np
from ensemble.preprocessing import randomizer
from keras.models import Model
from keras.layers import Dense,Input,Activation
from keras import regularizers
from keras.layers import normalization
from ensemble.tf_fusions import er_fusion
from keras.optimizers import SGD

dataset=Dataset(multilabel=True)
sample_number = 5000

pick=randomizer(sample_number)

input1=dataset.sources['cnn'][1]
input2=dataset.sources['mfcc'][1]
output=dataset.labels.shape[1]

features_1 = Input(shape=(input1,))
features_2 = Input(shape=(input2,))

e1 = Dense(4000, input_dim=input,activation='sigmoid',
                W_regularizer=regularizers.l2(l=0.))(features_1)
e1 = normalization.BatchNormalization()(e1)
f = Dense(2000,activation='sigmoid',
                W_regularizer=regularizers.l2(l=0.))(e1)
f = normalization.BatchNormalization()(f)

e2 = Dense(4000, input_dim=input,activation='sigmoid',
                W_regularizer=regularizers.l2(l=0.))(features_2)
e2 = normalization.BatchNormalization()(e2)
f1 = Dense(2000,activation='sigmoid',
                W_regularizer=regularizers.l2(l=0.))(e2)
f1 = normalization.BatchNormalization()(f1)

mfcc_out = Dense(output,activation='sigmoid',name="mfcc")(f)
cnn_out = Dense(output,activation='sigmoid',name="cnn")(f1)

out = Merge(mode=er_fusion,output_shape=(239,))([mfcc_out, cnn_out])
out = Activation('sigmoid')(out)

model = Model(input=[features_1,features_2], output=out)

model.compile(loss=objectives.binary_crossentropy,
                      optimizer=SGD(),
                      metrics=[metrics.categorical_accuracy])

model.summary()

cnn_x,\
cnn_y=next(dataset.generator(source='cnn', samples=sample_number, load_window=1, train=1))

cnn_x_t,\
cnn_y_t=next(dataset.generator(source='cnn', samples=sample_number, load_window=1, train=0))



mfcc_x,\
mfcc_y=next(dataset.generator(source='mfcc', samples=sample_number, load_window=1, train=1))

mfcc_x_t,\
mfcc_y_t=next(dataset.generator(source='mfcc', samples=sample_number, load_window=1, train=0))




check_stop = Checkpoint(validation_data=([cnn_x_t[pick],mfcc_x_t[pick]],cnn_y_t[pick]),
                                             previous_best=0.,
                                             verbose=1,mode='max',epochs_to_stop=30)



model.fit([cnn_x,mfcc_x], cnn_y, batch_size=128, nb_epoch=200, callbacks=[check_stop])


if hasattr(check_stop, 'best_weights'):
    model.set_weights(check_stop.best_weights)

y_p = model.predict([cnn_x_t,mfcc_x_t])
np.save("outputs/%s_output_MEAN_INCORPORATED_DNN_"%'mfcc_cnn',y_p)















