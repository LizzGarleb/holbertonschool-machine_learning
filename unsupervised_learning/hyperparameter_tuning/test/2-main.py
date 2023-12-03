#!/usr/bin/env python3

GP = __import__('2-gp').GaussianProcess
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    X_new = np.random.uniform(-np.pi, 2*np.pi, 1)
    print('X_new:', X_new)
    Y_new = f(X_new)
    print('Y_new:', Y_new)
    gp.update(X_new, Y_new)
    print(gp.X.shape, gp.X)
    print(gp.Y.shape, gp.Y)
    print(gp.K.shape, gp.K)

# Expected Output:
# X_new: [2.53931833]
# Y_new: [1.99720866]
# (3, 1) [[2.03085276]
#  [3.59890832]
#  [2.53931833]]
# (3, 1) [[ 0.92485357]
#  [-2.33925576]
#  [ 1.99720866]]
# (3, 3) [[4.         0.13150595 2.79327536]
#  [0.13150595 4.         0.84109203]
#  [2.79327536 0.84109203 4.        ]]