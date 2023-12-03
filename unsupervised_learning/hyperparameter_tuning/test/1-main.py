#!/usr/bin/env python3

GP = __import__('1-gp').GaussianProcess
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    X_s = np.random.uniform(-np.pi, 2*np.pi, (10, 1))
    mu, sig = gp.predict(X_s)
    print(mu.shape, mu)
    print(sig.shape, sig)

# Expected Output:
# (10,) [ 0.20148983  0.93469135  0.14512328 -0.99831012  0.21779183 -0.05063668
#  -0.00116747  0.03434981 -1.15092063  0.9221554 ]
# (10,) [1.90890408 0.01512125 3.91606789 2.42958747 3.81083574 3.99817545
#  3.99999903 3.9953012  3.05639472 0.37179608]