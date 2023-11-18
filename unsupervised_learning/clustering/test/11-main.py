#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
gmm = __import__('11-gmm').gmm

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)

    pi, m, S, clss, bic = gmm(X, 4)
    print(pi)
    print(m)
    print(S)
    print(bic)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(m[:, 0], m[:, 1], s=50, marker='*', c=list(range(4)))
    plt.show()

# Expected Output:
# [0.06054777 0.70467829 0.15613693 0.07863701]
# [[60.23478926 30.22428892]
#  [30.59365755 40.63753599]
#  [17.17946347 32.21805142]
#  [20.00762691 70.0169833 ]]
# [[[ 1.65286199e+01  1.02543709e-01]
#   [ 1.02543709e-01  1.65687019e+01]]

#  [[ 7.09781098e+01 -2.18081013e+00]
#   [-2.18081013e+00  7.78058769e+01]]

#  [[ 1.41559799e+02  7.97703830e+01]
#   [ 7.97703830e+01  6.39836725e+01]]

#  [[ 3.54632094e+01  1.12791052e+01]
#   [ 1.12791052e+01  3.21044161e+01]]]
# 189727.91411998263