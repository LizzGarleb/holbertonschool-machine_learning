#!/usr/bin/env python3

GP = __import__('2-gp').GaussianProcess
BO = __import__('3-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=2, sigma_f=3, xsi=0.05)
    print(bo.f is f)
    print(type(bo.gp) is GP)
    print(bo.gp.X is X_init)
    print(bo.gp.Y is Y_init)
    print(bo.gp.l)
    print(bo.gp.sigma_f)
    print(bo.X_s.shape, bo.X_s)
    print(bo.xsi)
    print(bo.minimize)

# Expected Output
# True
# True
# True
# True
# 2
# 3
# (50, 1) [[-3.14159265]
#  [-2.94925025]
#  [-2.75690784]
#  [-2.56456543]
#  [-2.37222302]
#  [-2.17988062]
#  [-1.98753821]
#  [-1.7951958 ]
#  [-1.60285339]
#  [-1.41051099]
#  [-1.21816858]
#  [-1.02582617]
#  [-0.83348377]
#  [-0.64114136]
#  [-0.44879895]
#  [-0.25645654]
#  [-0.06411414]
#  [ 0.12822827]
#  [ 0.32057068]
#  [ 0.51291309]
#  [ 0.70525549]
#  [ 0.8975979 ]
#  [ 1.08994031]
#  [ 1.28228272]
#  [ 1.47462512]
#  [ 1.66696753]
#  [ 1.85930994]
#  [ 2.05165235]
#  [ 2.24399475]
#  [ 2.43633716]
#  [ 2.62867957]
#  [ 2.82102197]
#  [ 3.01336438]
#  [ 3.20570679]
#  [ 3.3980492 ]
#  [ 3.5903916 ]
#  [ 3.78273401]
#  [ 3.97507642]
#  [ 4.16741883]
#  [ 4.35976123]
#  [ 4.55210364]
#  [ 4.74444605]
#  [ 4.93678846]
#  [ 5.12913086]
#  [ 5.32147327]
#  [ 5.51381568]
#  [ 5.70615809]
#  [ 5.89850049]
#  [ 6.0908429 ]
#  [ 6.28318531]]
# 0.05
# True