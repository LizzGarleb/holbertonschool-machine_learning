#!/usr/bin/env python3

import numpy as np

NN = __import__('9-neural_network').NeuralNetwork

lib_train = np.load('supervised_learning/classification/data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)
print(nn.A1)
print(nn.A2)
nn.A1 = 10
print(nn.A1)

# Expected Output:
# [[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
#   -1.34149673]
#  [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
#    0.07912172]
#  [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
#   -1.07836109]]
# [[0.]
#  [0.]
#  [0.]]
# [[ 1.06160017 -1.18488744 -1.80525169]]
# 0
# 0
# 0
# Traceback (most recent call last):
#   File "./9-main.py", line 19, in <module>
#     nn.A1 = 10
# AttributeError: can't set attribute