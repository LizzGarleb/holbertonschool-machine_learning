#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot
load_model = __import__('9-model').load_model
predict = __import__('13-predict').predict


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_test = datasets['X_test']
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = datasets['Y_test']

    network = load_model('network2.h5')
    Y_pred = predict(network, X_test)
    print(Y_pred)
    print(np.argmax(Y_pred, axis=1))
    print(Y_test)

# Expected Output:
# [[9.45855305e-08 1.23174982e-06 2.32663524e-06 ... 9.99959946e-01
#   8.68574972e-08 1.64004960e-05]
#  [1.32828617e-07 1.36074696e-05 9.99983311e-01 ... 1.28423885e-08
#   3.21557025e-07 1.41133481e-12]
#  [6.16613443e-06 9.99182522e-01 1.86337289e-04 ... 3.22229840e-04
#   1.28644533e-04 1.20764213e-07]
#  ...
#  [2.61130857e-11 6.05865438e-08 2.87002843e-12 ... 4.44421914e-07
#   2.17837393e-08 1.64936665e-07]
#  [2.04758184e-08 8.73848605e-10 3.49703627e-10 ... 3.04102166e-09
#   4.29783067e-05 7.22518101e-10]
#  [7.48594005e-08 1.35518596e-09 4.54492266e-08 ... 1.25057568e-11
#   2.57464561e-09 8.55768303e-11]]
# [7 2 1 ... 4 5 6]
# [7 2 1 ... 4 5 6]
