#!/usr/bin/env python3

import numpy as np

NN = __import__('10-neural_network').NeuralNetwork

lib_train = np.load('supervised_learning/classification/data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
nn._NeuralNetwork__b1 = np.ones((3, 1))
nn._NeuralNetwork__b2 = 1
A1, A2 = nn.forward_prop(X)
if A1 is nn.A1:
        print(A1)
if A2 is nn.A2:
        print(A2)

# Expected Output:
# [[5.34775247e-10 7.24627778e-04 4.52416436e-07 ... 8.75691930e-05
#   1.13141966e-06 6.55799932e-01]
#  [9.99652394e-01 9.99999995e-01 6.77919152e-01 ... 1.00000000e+00
#   9.99662771e-01 9.99990554e-01]
#  [5.57969669e-01 2.51645047e-02 4.04250047e-04 ... 1.57024117e-01
#   9.97325173e-01 7.41310459e-02]]
# [[0.23294587 0.44286405 0.54884691 ... 0.38502756 0.12079644 0.593269  ]]