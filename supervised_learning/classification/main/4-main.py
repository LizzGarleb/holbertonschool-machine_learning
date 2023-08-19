#!/usr/bin/env python3

import numpy as np

Neuron = __import__('4-neuron').Neuron

lib_train = np.load('supervised_learning/classification/data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A, cost = neuron.evaluate(X, Y)
print(A)
print(cost)

# Expected Output:
# [[0 0 0 ... 0 0 0]]
# 4.365104944262272