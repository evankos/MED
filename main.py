import ensemble
from ensemble.classifiers import Dnn
from ensemble.topology import Parallel

classifier=Dnn(10,1,name='DNN1')


parallel_ensemble=Parallel([classifier])

parallel_ensemble.add(Dnn(10,1,name='DNN2'))
parallel_ensemble.save_models()
parallel_ensemble.save_weights()