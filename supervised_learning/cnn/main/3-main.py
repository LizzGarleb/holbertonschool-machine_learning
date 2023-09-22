#!/usr/bin/env python3

import numpy as np
pool_backward = __import__('3-pool_backward').pool_backward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    _, h, w = X_train.shape
    X_train_a = X_train[:10].reshape((-1, h, w, 1))
    X_train_b = 1 - X_train_a
    X_train_c = np.concatenate((X_train_a, X_train_b), axis=3)

    dA = np.random.randn(10, h // 3, w // 3, 2)
    print(pool_backward(dA, X_train_c, (3, 3), stride=(3, 3)))

# Expected Output:
# Beginning
# [[[[ 1.76405235  0.40015721]
#    [ 1.76405235  0.40015721]
#    [ 0.97873798  2.2408932 ]
#    ...
#    [ 2.26975462 -1.45436567]
#    [ 0.04575852 -0.18718385]
#    [ 0.04575852 -0.18718385]]

# Ending
#    [[-0.92196766  1.77405634]
#    [-0.92196766  1.77405634]
#    [ 0.02875624  0.55296385]
#    ...
#    [ 0.24288982 -0.40083471]
#    [-1.02155985 -0.47002432]
#    [-1.02155985 -0.47002432]]]]