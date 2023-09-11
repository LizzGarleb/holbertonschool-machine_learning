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
train_model = __import__('7-train').train_model 

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
    epochs = 1000
    train_model(network, X_train, Y_train_oh, batch_size, epochs,
                validation_data=(X_valid, Y_valid_oh), early_stopping=True,
                patience=3, learning_rate_decay=True, alpha=alpha)

# Expected Output:
# Epoch 1/1000

# Epoch 00001: LearningRateScheduler setting learning rate to 0.001.
# 782/782 [==============================] - 5s 4ms/step - loss: 0.3536 - acc: 0.9205 - val_loss: 0.2088 - val_acc: 0.9639
# Epoch 2/1000

# Epoch 00002: LearningRateScheduler setting learning rate to 0.0005.
# 782/782 [==============================] - 3s 4ms/step - loss: 0.1803 - acc: 0.9703 - val_loss: 0.1642 - val_acc: 0.9744
# Epoch 3/1000

# Epoch 00003: LearningRateScheduler setting learning rate to 0.0003333333333333333.
# 782/782 [==============================] - 3s 4ms/step - loss: 0.1447 - acc: 0.9800 - val_loss: 0.1522 - val_acc: 0.9768
# Epoch 4/1000

# Epoch 00004: LearningRateScheduler setting learning rate to 0.00025.
# 782/782 [==============================] - 3s 4ms/step - loss: 0.1262 - acc: 0.9844 - val_loss: 0.1449 - val_acc: 0.9790
# Epoch 5/1000

# ...

# Epoch 00043: LearningRateScheduler setting learning rate to 2.3255813953488374e-05.
# 782/782 [==============================] - 3s 3ms/step - loss: 0.0564 - acc: 0.9990 - val_loss: 0.1059 - val_acc: 0.9827
# Epoch 44/1000

# Epoch 00044: LearningRateScheduler setting learning rate to 2.272727272727273e-05.
# 782/782 [==============================] - 3s 3ms/step - loss: 0.0561 - acc: 0.9988 - val_loss: 0.1061 - val_acc: 0.9829
# Epoch 45/1000

# Epoch 00045: LearningRateScheduler setting learning rate to 2.2222222222222223e-05.
# 782/782 [==============================] - 3s 3ms/step - loss: 0.0559 - acc: 0.9988 - val_loss: 0.1063 - val_acc: 0.9828 
