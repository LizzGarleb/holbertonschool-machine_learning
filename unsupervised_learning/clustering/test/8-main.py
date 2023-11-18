#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    k = 4
    pi, m, S, g, l = expectation_maximization(X, k, 150, verbose=True)
    clss = np.sum(g * np.arange(k).reshape(k, 1), axis=0)
    plt.scatter(X[:, 0], X[:, 1], s=20, c=clss)
    plt.scatter(m[:, 0], m[:, 1], s=50, c=np.arange(k), marker='*')
    plt.show()
    print(X.shape[0] * pi)
    print(m)
    print(S)
    print(l)

# Expection Output:
# Log Likelihood after 0 iterations: -652797.78665
# Log Likelihood after 10 iterations: -94855.45662
# Log Likelihood after 20 iterations: -94714.52057
# Log Likelihood after 30 iterations: -94590.87362
# Log Likelihood after 40 iterations: -94440.40559
# Log Likelihood after 50 iterations: -94439.93891
# Log Likelihood after 52 iterations: -94439.93889

# [ 761.03239903  747.62391034 1005.60275934 9985.74093129]
# [[60.18888335 30.19707607]
#  [ 5.05794926 24.92588821]
#  [20.03438453 69.84721009]
#  [29.89607379 40.12519148]]
# [[[16.85183426  0.2547388 ]
#   [ 0.2547388  16.49432111]]

#  [[15.19555672  9.62661086]
#   [ 9.62661086 15.47295413]]

#  [[35.58332494 11.08419454]
#   [11.08419454 33.09463207]]

#  [[74.52083678  5.20755533]
#   [ 5.20755533 73.87299705]]]
# -94439.93889004056