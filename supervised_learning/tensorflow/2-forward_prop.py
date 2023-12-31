#!/usr/bin/env python3
"""
module forward_prop
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network
    """
    layer = create_layer(x, layer_sizes[0], activations[0])
    for idx in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[idx], activations[idx])
    return layer
