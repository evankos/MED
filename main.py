import ensemble
from ensemble.classifiers import Dnn,Svm
from ensemble.topology import Parallel
import numpy as np

from corpus.common import mfcc_file

#test data
x=np.random.rand(5000,3)
y=np.zeros((5000,10))
categories=np.random.randint(10, size=5000)
y[np.arange(5000),categories]=1


parallel_ensemble=Parallel()

parallel_ensemble.add(Dnn(3,10,name='DNN1'))
parallel_ensemble.add(Dnn(3,10,name='DNN2'))
parallel_ensemble.add(Svm(name='SVM1'))
parallel_ensemble.compile(fusion='avg')

# parallel_ensemble.train_on_batch([[x, y], [x, y]])
parallel_ensemble.fit([[x, y], [x, y], [x, y]])
print(parallel_ensemble.histories)

#single prediction after late fusion
print(parallel_ensemble.predict([x,x,x]).shape)

parallel_ensemble.save_models()
parallel_ensemble.save_weights()





