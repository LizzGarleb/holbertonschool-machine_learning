#!/usr/bin/env python3

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
build_model = __import__('1-input').build_model

if __name__ == '__main__':
    network = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    network.summary()
    print(network.losses)

# Expected Output:
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         [(None, 784)]             0         
# _________________________________________________________________
# dense (Dense)                (None, 256)               200960    
# _________________________________________________________________
# dropout (Dropout)            (None, 256)               0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 256)               65792     
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 256)               0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                2570      
# =================================================================
# Total params: 269,322
# Trainable params: 269,322
# Non-trainable params: 0
# _________________________________________________________________
# [<tf.Tensor: shape=(), dtype=float32, numpy=0.5120259>, <tf.Tensor: shape=(), 
# dtype=float32, numpy=0.5118243>, <tf.Tensor: shape=(), dtype=float32, numpy=0.020181004>]