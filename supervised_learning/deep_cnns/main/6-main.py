#!/usr/bin/env python3

import tensorflow.keras as K
transition_layer = __import__('6-transition_layer').transition_layer

if __name__ == '__main__':
    X = K.Input(shape=(56, 56, 256))
    Y, nb_filters = transition_layer(X, 256, 0.5)
    model = K.models.Model(inputs=X, outputs=Y)
    model.summary()
    print(nb_filters)

# Expected Output:
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         [(None, 56, 56, 256)]     0         
# _________________________________________________________________
# batch_normalization (BatchNo (None, 56, 56, 256)       1024      
# _________________________________________________________________
# activation (Activation)      (None, 56, 56, 256)       0         
# _________________________________________________________________
# conv2d (Conv2D)              (None, 56, 56, 128)       32896     
# _________________________________________________________________
# average_pooling2d (AveragePo (None, 28, 28, 128)       0         
# =================================================================
# Total params: 33,920
# Trainable params: 33,408
# Non-trainable params: 512
# _________________________________________________________________
# 128
