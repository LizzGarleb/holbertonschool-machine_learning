#!/usr/bin/env python3

import numpy as np
dropout_forward_prop = __import__('4-dropout_forward_prop').dropout_forward_prop
dropout_gradient_descent = __import__('5-dropout_gradient_descent').dropout_gradient_descent


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

    cache = dropout_forward_prop(X_train, weights, 3, 0.8)
    print(weights['W2'])
    dropout_gradient_descent(Y_train_oh, weights, cache, 0.1, 0.8, 3)
    print(weights['W2'])

# Expected Output:
# [[-1.9282086  -0.71324613 -1.33191318 ... -2.14202626 -0.07737407
#    0.99832167]
#  [-0.0237149  -0.18364778  0.08337452 ... -0.06093055 -0.03924408
#   -2.17625294]
#  [-0.16181888  0.49237435 -0.47196279 ...  0.97504077  0.16272698
#    0.56159916]
#  ...
#  [ 0.39842474 -0.09870005  1.32173992 ... -0.33210834  0.66215988
#    0.87211421]
#  [ 0.15767221  0.42236212  1.004765   ...  0.69883284  0.70857088
#   -0.44427252]
#  [ 2.68588811 -0.60351958 -1.0759598  ... -1.2437044   0.69462324
#    1.00090403]]
# [[-1.92044686 -0.71894673 -1.32811693 ... -2.14071955 -0.07158198
#    0.98206832]
#  [-0.03706116 -0.17088483  0.07798748 ... -0.07245569 -0.0491215
#   -2.16245276]
#  [-0.17198668  0.49842244 -0.47369328 ...  0.96880194  0.15497217
#    0.5693131 ]
#  ...
#  [ 0.41997262 -0.11452751  1.32873227 ... -0.31312321  0.67162237
#    0.85928296]
#  [ 0.13702353  0.44237056  1.00139188 ...  0.68128208  0.69020934
#   -0.43055442]
#  [ 2.66514017 -0.59204122 -1.08943163 ... -1.26238074  0.69280683
#    1.02353101]]