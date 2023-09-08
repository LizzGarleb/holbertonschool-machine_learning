#!/usr/bin/env python3

import numpy as np
l2_reg_gradient_descent = __import__('1-l2_reg_gradient_descent').l2_reg_gradient_descent


def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    Y_train_oh = one_hot(Y_train, 10)

    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['b1'] = np.zeros((256, 1))
    weights['W2'] = np.random.randn(128, 256)
    weights['b2'] = np.zeros((128, 1))
    weights['W3'] = np.random.randn(10, 128)
    weights['b3'] = np.zeros((10, 1))

    cache = {}
    cache['A0'] = X_train
    cache['A1'] = np.tanh(np.matmul(weights['W1'], cache['A0']) + weights['b1'])
    cache['A2'] = np.tanh(np.matmul(weights['W2'], cache['A1']) + weights['b2'])
    Z3 = np.matmul(weights['W3'], cache['A2']) + weights['b3']
    cache['A3'] = np.exp(Z3) / np.sum(np.exp(Z3), axis=0)
    print(weights['W1'])
    l2_reg_gradient_descent(Y_train_oh, weights, cache, 0.1, 0.1, 3)
    print(weights['W1'])

# Expected Output:
# [[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
#   -1.34149673]
#  [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
#    0.07912172]
#  [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
#   -1.07836109]
#  ...
#  [-0.60467085  0.54751161 -1.23317415 ...  0.82895532  1.44161136
#    0.18972404]
#  [-0.41044606  0.85719512  0.71789835 ... -0.73954771  0.5074628
#    1.23022874]
#  [ 0.43129249  0.60767018 -0.07749988 ... -0.26611561  2.52287972
#    0.73131543]]
# [[ 1.76405199  0.40015713  0.97873779 ...  0.52130364  0.61192707
#   -1.34149646]
#  [ 0.47689827  0.14844955  0.52904513 ...  0.09600419 -0.04511329
#    0.07912171]
#  [ 0.85053051 -0.83912402 -1.01177388 ... -0.07223874  0.31112438
#   -1.07836088]
#  ...
#  [-0.60467073  0.5475115  -1.2331739  ...  0.82895516  1.44161107
#    0.189724  ]
#  [-0.41044598  0.85719495  0.71789821 ... -0.73954756  0.5074627
#    1.2302285 ]
#  [ 0.4312924   0.60767006 -0.07749987 ... -0.26611556  2.52287922
#    0.73131529]]
