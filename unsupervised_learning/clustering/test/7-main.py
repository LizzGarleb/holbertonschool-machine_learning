#!/usr/bin/env python3

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    pi, m, S = initialize(X, 4)
    g, _ = expectation(X, pi, m, S)
    pi, m, S = maximization(X, g)
    print(pi)
    print(m)
    print(S)

# Expected Output:
# [0.10104901 0.24748822 0.1193333  0.53212947]
# [[54.7440558  31.80888393]
#  [16.84099873 31.20560148]
#  [21.42588061 65.51441875]
#  [32.33208369 41.80830251]]
# [[[64.05063663 -2.13941814]
#   [-2.13941814 41.90354928]]

#  [[72.72404579  9.96322554]
#   [ 9.96322554 53.05035303]]

#  [[46.20933259  1.08979413]
#   [ 1.08979413 66.9841323 ]]

#  [[35.04054823 -0.94790014]
#   [-0.94790014 45.14948772]]]