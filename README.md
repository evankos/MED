# MED: Multimedia Event Detection

## Getting started

This repository holds an ensemble of multi-modal classifiers developed within
the frame of a research project. The framework relies heavily on Keras and also
tries to mirror its structure for maintainability reasons.

Here is a simple initiation of an ensemble of parallel classifiers:

```python
from ensemble.classifiers import Dnn
from ensemble.topology import Parallel

parallel_ensemble=Parallel()
```

Note its very similar to the implementation of the `Sequential` model in Keras.

Stacking classifiers with `add()`:

```python
parallel_ensemble.add(Dnn(3,10,name='DNN1'))
parallel_ensemble.add(Dnn(3,10,name='DNN2'))
```
Compile and build the classifiers and the fusion code:
```python
parallel_ensemble.compile(fusion='avg')
```
Train the all the ensemble:
```python
parallel_ensemble.fit([[x, y], [x, y]])
```
Retrieve single prediction after the late fusion of your choice:
```python
print(parallel_ensemble.predict([x,x]))
```
Saving the model structure as JSON and the weights as MD5:
```python
parallel_ensemble.save_models()
parallel_ensemble.save_weights()
```
