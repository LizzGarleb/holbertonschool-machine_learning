#!/usr/bin/env python3
"""
Deep CNNs Module
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block as described in Densely Connected Convolutional
    Networks
    """
    init = K.initializers.he_normal()
    for i in range(layers):
        batch1 = K.layers.BatchNormalization()(X)
        relu1 = K.layers.Activation('relu')(batch1)
        bottleneck = K.layers.Conv2D(filters=4*growth_rate, kernel_size=1,
                                     padding='same',
                                     kernel_initializer=init)(relu1)
        batch2 = K.layers.BatchNormalization()(bottleneck)
        relu2 = K.layers.Activation('relu')(batch2)
        X_conv = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                 padding='same',
                                 kernel_initializer=init)(relu2)
        X = K.layers.concatenate([X, X_conv])
        nb_filters += growth_rate
    return X, nb_filters
