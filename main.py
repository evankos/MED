import ensemble
from ensemble.classifiers import Dnn
from ensemble.topology import Parallel
import numpy as np
from corpus.common import mfcc_file

#test data
x=np.random.rand(128,3)
y=np.zeros((128,10))
y[np.arange(128),np.random.randint(10, size=128)]=1


parallel_ensemble=Parallel()

parallel_ensemble.add(Dnn(3,10,name='DNN1'))
parallel_ensemble.add(Dnn(3,10,name='DNN2'))
parallel_ensemble.compile(fusion='avg')

# parallel_ensemble.train_on_batch([[x, y], [x, y]])
parallel_ensemble.fit([[x, y], [x, y]])
print(parallel_ensemble.histories)

#single prediction after late fusion
print(parallel_ensemble.predict([x,x]))

parallel_ensemble.save_models()
parallel_ensemble.save_weights()





