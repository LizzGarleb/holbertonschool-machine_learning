#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
BIC = __import__('9-BIC').BIC

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    best_k, best_result, l, b = BIC(X, kmin=1, kmax=10)
    print(best_k)
    print(best_result)
    print(l)
    print(b)
    x = np.arange(1, 11)
    plt.plot(x, l, 'r')
    plt.xlabel('Clusters')
    plt.ylabel('Log Likelihood')
    plt.tight_layout()
    plt.show()
    plt.plot(x, b, 'b')
    plt.xlabel('Clusters')
    plt.ylabel('BIC')
    plt.tight_layout()
    plt.show()

# Expected Output:
# 4
# (array([0.79885962, 0.08044842, 0.06088258, 0.05980938]), array([[29.89606417, 40.12518027],
#        [20.0343883 , 69.84718588],
#        [60.18888407, 30.19707372],
#        [ 5.05788987, 24.92583792]]), array([[[74.52101284,  5.20770764],
#         [ 5.20770764, 73.8729309 ]],

#        [[35.58334497, 11.08416742],
#         [11.08416742, 33.09483747]],

#        [[16.85183256,  0.25475122],
#         [ 0.25475122, 16.4943092 ]],

#        [[15.19520213,  9.62633552],
#         [ 9.62633552, 15.47268905]]]))
# [-98801.40298366 -96729.95558846 -95798.40406023 -94439.93888882
#  -94435.87750008 -94428.62217176 -94426.71159745 -94425.5860871
#  -94421.41864281 -94416.43390835]
# [197649.97338694 193563.67950008 191757.17734716 189096.84790787
#  189145.32603394 189187.41628084 189240.19603576 189294.54591859
#  189342.81193356 189389.44336818]
