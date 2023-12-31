#!/usr/bin/env python3

import numpy as np

Deep = __import__('17-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('supervised_learning/classification/data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print(deep.L)

# Expected Output:
# {}
# {'b1': array([[0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.]]), 'b2': array([[0.],
#        [0.],
#        [0.]]), 'W2': array([[ 0.4609219 ,  0.56004008, -1.2250799 , -0.09454199,  0.57799141],
#        [-0.16310703,  0.06882082, -0.94578088, -0.30359994,  1.15661914],
#        [-0.49841799, -0.9111359 ,  0.09453424,  0.49877298,  0.75503205]]), 'W1': array([[ 0.0890981 ,  0.02021099,  0.04943373, ...,  0.02632982,
#          0.03090699, -0.06775582],
#        [ 0.02408701,  0.00749784,  0.02672082, ...,  0.00484894,
#         -0.00227857,  0.00399625],
#        [ 0.04295829, -0.04238217, -0.05110231, ..., -0.00364861,
#          0.01571416, -0.05446546],
#        [ 0.05361891, -0.05984585, -0.09117898, ..., -0.03094292,
#         -0.01925805, -0.06308145],
#        [-0.01667953, -0.04216413,  0.06239623, ..., -0.02024521,
#         -0.05159656, -0.02373981]]), 'b3': array([[0.]]), 'W3': array([[-0.42271877,  0.18165055,  0.4444639 ]])}
# 3
# Traceback (most recent call last):
#   File "./17-main.py", line 16, in <module>
#     deep.L = 10
# AttributeError: can't set attribute