#!/usr/bin/env python3

import numpy as np
pdf = __import__('5-pdf').pdf

if __name__ == '__main__':
    np.random.seed(0)
    m = np.array([12, 30, 10])
    S = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    X = np.random.multivariate_normal(m, S, 10000)
    P = pdf(X, m, S)
    print(P)

# Expected Output:
# [3.47450910e-05 2.53649178e-06 1.80348301e-04 ... 1.24604061e-04
#  1.86345129e-04 2.59397003e-05]