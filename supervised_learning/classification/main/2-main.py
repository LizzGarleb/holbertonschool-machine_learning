#!/usr/bin/env python3

import numpy as np

Neuron = __import__('2-neuron').Neuron

lib_train = np.load('supervised_learning/classification/data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
neuron._Neuron__b = 1
A = neuron.forward_prop(X)
if (A is neuron.A):
        print(A)

# Expected Output:
# [[5.34775247e-10 7.24627778e-04 4.52416436e-07 ... 8.75691930e-05
#   1.13141966e-06 6.55799932e-01]]