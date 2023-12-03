#!/usr/bin/env python3

BO = __import__('4-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6, sigma_f=2, xsi=0.05)
    X_next, EI = bo.acquisition()

    print(EI)
    print(X_next)

    plt.scatter(X_init.reshape(-1), Y_init.reshape(-1), color='g')
    plt.plot(bo.X_s.reshape(-1), EI.reshape(-1), color='r')
    plt.axvline(x=X_next)
    plt.show()

# Expected Output:
# [6.77642382e-01 6.77642382e-01 6.77642382e-01 6.77642382e-01
#  6.77642382e-01 6.77642382e-01 6.77642382e-01 6.77642382e-01
#  6.77642379e-01 6.77642362e-01 6.77642264e-01 6.77641744e-01
#  6.77639277e-01 6.77628755e-01 6.77588381e-01 6.77448973e-01
#  6.77014261e-01 6.75778547e-01 6.72513223e-01 6.64262238e-01
#  6.43934968e-01 5.95940851e-01 4.93763541e-01 3.15415142e-01
#  1.01026267e-01 1.73225936e-03 4.29042673e-28 0.00000000e+00
#  4.54945116e-13 1.14549081e-02 1.74765619e-01 3.78063126e-01
#  4.19729153e-01 2.79303426e-01 7.84942221e-02 0.00000000e+00
#  8.33323492e-02 3.25320033e-01 5.70580150e-01 7.20239593e-01
#  7.65975535e-01 7.52693111e-01 7.24099594e-01 7.01220863e-01
#  6.87941196e-01 6.81608621e-01 6.79006118e-01 6.78063616e-01
#  6.77759591e-01 6.77671794e-01]
# [4.55210364]