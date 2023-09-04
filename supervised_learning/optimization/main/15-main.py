#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
model = __import__('15-model').model

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)
    X_valid_3D = lib['X_valid']
    Y_valid = lib['Y_valid']
    X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
    Y_valid_oh = one_hot(Y_valid, 10)

    layer_sizes = [256, 256, 10]
    activations = [tf.nn.tanh, tf.nn.tanh, None]

    np.random.seed(0)
    tf.set_random_seed(0)
    save_path = model((X_train, Y_train_oh), (X_valid, Y_valid_oh), layer_sizes,
                                 activations, save_path='./model.ckpt')
    print('Model saved in path: {}'.format(save_path))

# Expected Output:
# After 0 epochs:
#     Training Cost: 2.5810317993164062
#     Training Accuracy: 0.16808000206947327
#     Validation Cost: 2.5596187114715576
#     Validation Accuracy: 0.16859999299049377
#     Step 100:
#         Cost: 0.297500342130661
#         Accuracy 0.90625
#     Step 200:
#         Cost: 0.27544915676116943
#         Accuracy 0.875

#     ...

#     Step 1500:
#         Cost: 0.09414251148700714
#         Accuracy 1.0
# After 1 epochs:
#     Training Cost: 0.13064345717430115
#     Training Accuracy: 0.9625800251960754
#     Validation Cost: 0.14304184913635254
#     Validation Accuracy: 0.9595000147819519

# ...

# After 4 epochs:
#     Training Cost: 0.03584253042936325
#     Training Accuracy: 0.9912999868392944
#     Validation Cost: 0.0853486955165863
#     Validation Accuracy: 0.9750999808311462
#     Step 100:
#         Cost: 0.03150765225291252
#         Accuracy 1.0
#     Step 200:
#         Cost: 0.020879564806818962
#         Accuracy 1.0

#     ...

#     Step 1500:
#         Cost: 0.015160675160586834
#         Accuracy 1.0
# After 5 epochs:
#     Training Cost: 0.025094907730817795
#     Training Accuracy: 0.9940199851989746
#     Validation Cost: 0.08191727101802826
#     Validation Accuracy: 0.9750999808311462
# Model saved in path: ./model.ckpt