#!/usr/bin/env python3
""" Module Attention """
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculate the scaled dot product attention
    """
    q = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_q = q / tf.math.sqrt(dk)
    if mask is not None:
        scaled_q += (mask * -1e9)
    weights = tf.nn.softmax(scaled_q, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
