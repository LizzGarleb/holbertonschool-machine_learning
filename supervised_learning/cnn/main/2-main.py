#!/usr/bin/env python3

import numpy as np
conv_backward = __import__('2-conv_backward').conv_backward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    _, h, w = X_train.shape
    X_train_c = X_train[:10].reshape((-1, h, w, 1))

    W = np.random.randn(3, 3, 1, 2)
    b = np.random.randn(1, 1, 1, 2)

    dZ = np.random.randn(10, h - 2, w - 2, 2)
    print(conv_backward(dZ, X_train_c, W, b, padding="valid"))

# Expected Output:
# Beginning
# (array([[[[-4.24205748],
#          [ 0.19390938],
#          [-2.80168847],
#          ...,
#          [-2.93059274],
#          [-0.74257184],
#          [ 1.23556676]],

# Ending
#          [-0.28586324],
#          [ 2.24643738],
#          [ 0.74045003]]]]), array([[[[ 10.13352674, -25.15674655]],
#         [[ 33.27872337, -64.99062958]],
#         [[ 31.29539025, -77.29275492]]],
#        [[[ 10.61025981, -31.7337223 ]],
#         [[ 10.34048231, -65.19271124]],
#         [[ -1.73024336, -76.98703808]]],
#        [[[ -1.49204439, -33.46094911]],
#         [[  4.04542976, -63.47295685]],
#         [[  2.9243666 , -64.29296016]]]]), array([[[[-113.18404846, -121.902714  ]]]]))
