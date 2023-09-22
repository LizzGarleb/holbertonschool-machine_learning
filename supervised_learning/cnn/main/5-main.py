#!/usr/bin/env python3
"""
Main file
"""
# Force Seed - fix for Keras
SEED = 0
import matplotlib.pyplot as plt
import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
import tensorflow.keras as K

lenet5 = __import__('5-lenet5').lenet5

if __name__ == "__main__":
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    m, h, w = X_train.shape
    X_train_c = X_train.reshape((-1, h, w, 1))
    Y_train = lib['Y_train']
    Y_train_oh = K.utils.to_categorical(Y_train, num_classes=10)
    X_valid = lib['X_valid']
    X_valid_c = X_valid.reshape((-1, h, w, 1))
    Y_valid = lib['Y_valid']
    Y_valid_oh = K.utils.to_categorical(Y_valid, num_classes=10)
    X = K.Input(shape=(h, w, 1))
    model = lenet5(X)
    batch_size = 32
    epochs = 5
    model.fit(X_train_c, Y_train_oh, batch_size=batch_size, epochs=epochs,
                       validation_data=(X_valid_c, Y_valid_oh))
    Y_pred = model.predict(X_valid_c)
    print(Y_pred[0])
    Y_pred = np.argmax(Y_pred, 1)
    plt.imshow(X_valid[0])
    plt.title(str(Y_valid[0]) + ' : ' + str(Y_pred[0]))
    plt.show()

# Expected Output:
# Epoch 1/5
# 1563/1563 [==============================] - 11s 4ms/step - loss: 0.1665 - accuracy: 0.9489 - val_loss: 0.0596 - val_accuracy: 0.9813
# Epoch 2/5
# 1563/1563 [==============================] - 6s 4ms/step - loss: 0.0594 - accuracy: 0.9820 - val_loss: 0.0489 - val_accuracy: 0.9859
# Epoch 3/5
# 1563/1563 [==============================] - 6s 4ms/step - loss: 0.0408 - accuracy: 0.9869 - val_loss: 0.0469 - val_accuracy: 0.9870
# Epoch 4/5
# 1563/1563 [==============================] - 6s 4ms/step - loss: 0.0346 - accuracy: 0.9889 - val_loss: 0.0482 - val_accuracy: 0.9870
# Epoch 5/5
# 1563/1563 [==============================] - 6s 4ms/step - loss: 0.0255 - accuracy: 0.9917 - val_loss: 0.0483 - val_accuracy: 0.9875
# [4.5772337e-16 2.6305156e-12 1.4343354e-13 1.0000000e+00 2.8758866e-17
#  3.4095468e-08 3.7155215e-15 3.2845108e-13 3.5915697e-11 4.5209600e-11]