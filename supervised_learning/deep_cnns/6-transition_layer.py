#!/usr/bin/env python3
"""
Deep CNNs Module
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    builds a transition layer as described in Densely Connected Convolutional
    Networks
    """
    init = K.initializers.he_normal()
    nfilter = int(nb_filters * compression)
    batch1 = K.layers.BatchNormalization()(X)
    relu1 = K.layers.Activation('relu')(batch1)
    conv = K.layers.Conv2D(filters=nfilter, kernel_size=1, padding='same',
                           kernel_initializer=init)(relu1)
    avg_pool = K.layers.AveragePooling2D(pool_size=2, strides=2,
                                         padding='same')(conv)
    return avg_pool, nfilter
