#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Neuron = __import__('7-neuron').Neuron

lib_train = np.load('supervised_learning/classification/data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

np.random.seed(1)
neuron = Neuron(X_train.shape[0])
neuron.train(X_train, Y_train, iterations=1000)

# Expect Output:
# Cost after 0 iterations: 3.8989412337124083
# Cost after 100 iterations: 0.2232718315285948
# Cost after 200 iterations: 0.10540576545530819
# Cost after 300 iterations: 0.0706159342381332
# Cost after 400 iterations: 0.05413237578864413
# Cost after 500 iterations: 0.04456270794024501
# Cost after 600 iterations: 0.03834941270694736
# Cost after 700 iterations: 0.03401018069916897
# Cost after 800 iterations: 0.03080947371051137
# Cost after 900 iterations: 0.02834594904169237
# Cost after 1000 iterations: 0.02638467072536922