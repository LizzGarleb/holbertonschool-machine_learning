#!/usr/bin/env python3

BO = __import__('5-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6, sigma_f=2)
    X_opt, Y_opt = bo.optimize(50)
    print('Optimal X:', X_opt)
    print('Optimal Y:', Y_opt)
    print('All sample inputs:', bo.gp.X)

# Expected Output:
# Optimal X: [0.8975979]
# Optimal Y: [-2.92478374]
# All sample inputs: [[ 2.03085276]
#  [ 3.59890832]
#  [ 4.55210364]
#  [ 5.89850049]
#  [-3.14159265]
#  [-0.83348377]
#  [ 0.70525549]
#  [-2.17988062]
#  [ 3.01336438]
#  [ 3.97507642]
#  [ 1.28228272]
#  [ 5.12913086]
#  [ 0.12822827]
#  [ 6.28318531]
#  [-1.60285339]
#  [-2.75690784]
#  [-2.56456543]
#  [ 0.8975979 ]
#  [ 2.43633716]
#  [-0.44879895]]