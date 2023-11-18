#!/usr/bin/env python3

import numpy as np
pca = __import__('1-pca').pca

X = np.loadtxt("mnist2500_X.txt")
print('X:', X.shape)
print(X)
T = pca(X, 50)
print('T:', T.shape)
print(T)

# Expected Output: 
# X: (2500, 784)
# [[1. 1. 1. ... 1. 1. 1.]
#  [1. 1. 1. ... 1. 1. 1.]
#  [1. 1. 1. ... 1. 1. 1.]
#  ...
#  [1. 1. 1. ... 1. 1. 1.]
#  [1. 1. 1. ... 1. 1. 1.]
#  [1. 1. 1. ... 1. 1. 1.]]
# T: (2500, 50)
# [[-0.61344587  1.37452188 -1.41781926 ... -0.42685217  0.02276617
#    0.1076424 ]
#  [-5.00379081  1.94540396  1.49147124 ...  0.26249077 -0.4134049
#   -1.15489853]
#  [-0.31463237 -2.11658407  0.36608266 ... -0.71665401 -0.18946283
#    0.32878802]
#  ...
#  [ 3.52302175  4.1962009  -0.52129062 ... -0.24412645  0.02189273
#    0.19223197]
#  [-0.81387035 -2.43970416  0.33244717 ... -0.55367626 -0.64632309
#    0.42547833]
#  [-2.25717018  3.67177791  2.83905021 ... -0.35014766 -0.01807652
#    0.31548087]]