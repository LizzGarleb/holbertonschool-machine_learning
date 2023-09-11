#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
import tensorflow.keras as K

# Imports
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('6-train').train_model


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    batch_size = 64
    epochs = 30
    train_model(network, X_train, Y_train_oh, batch_size, epochs,
                validation_data=(X_valid, Y_valid_oh), early_stopping=True,
                patience=3)
    
# Expected Output:
# Epoch 1/30
# 782/782 [==============================] - 5s 4ms/step - loss: 0.3536 - acc: 0.9205 - val_loss: 0.2088 - val_acc: 0.9639
# Epoch 2/30
# 782/782 [==============================] - 3s 3ms/step - loss: 0.1949 - acc: 0.9658 - val_loss: 0.1681 - val_acc: 0.9726
# Epoch 3/30
# 782/782 [==============================] - 3s 3ms/step - loss: 0.1574 - acc: 0.9758 - val_loss: 0.1637 - val_acc: 0.9741
# Epoch 4/30
# 782/782 [==============================] - 3s 3ms/step - loss: 0.1369 - acc: 0.9804 - val_loss: 0.1562 - val_acc: 0.9760
# Epoch 5/30
# 782/782 [==============================] - 3s 3ms/step - loss: 0.1257 - acc: 0.9834 - val_loss: 0.1585 - val_acc: 0.9751
# Epoch 6/30
# 782/782 [==============================] - 3s 3ms/step - loss: 0.1151 - acc: 0.9857 - val_loss: 0.1503 - val_acc: 0.9773
# Epoch 7/30
# 782/782 [==============================] - 3s 3ms/step - loss: 0.1100 - acc: 0.9866 - val_loss: 0.1500 - val_acc: 0.9760
# Epoch 8/30
# 782/782 [==============================] - 3s 3ms/step - loss: 0.1071 - acc: 0.9875 - val_loss: 0.1395 - val_acc: 0.9790
# Epoch 9/30
# 782/782 [==============================] - 3s 3ms/step - loss: 0.1015 - acc: 0.9889 - val_loss: 0.1406 - val_acc: 0.9787
# Epoch 10/30
# 782/782 [==============================] - 3s 3ms/step - loss: 0.1004 - acc: 0.9883 - val_loss: 0.1459 - val_acc: 0.9773
# Epoch 11/30
# 782/782 [==============================] - 3s 3ms/step - loss: 0.0943 - acc: 0.9907 - val_loss: 0.1477 - val_acc: 0.9786
