#!/usr/bin/env python3

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    pi, m, S = initialize(X, 4)
    g, l = expectation(X, pi, m, S)
    print(g)
    print(np.sum(g, axis=0))
    print(l)

# Expected Output:
# [[1.98542668e-055 1.00000000e+000 1.56526421e-185 ... 1.00000000e+000
#   3.70567311e-236 1.91892348e-012]
#  [6.97883333e-085 2.28658376e-279 9.28518983e-065 ... 8.12227631e-287
#   1.53690661e-032 3.17417182e-181]
#  [9.79811365e-234 2.28658376e-279 2.35073465e-095 ... 1.65904890e-298
#   9.62514613e-068 5.67072057e-183]
#  [1.00000000e+000 7.21133039e-186 1.00000000e+000 ... 2.42138447e-125
#   1.00000000e+000 1.00000000e+000]]
# [1. 1. 1. ... 1. 1. 1.]
# -652797.7866541843