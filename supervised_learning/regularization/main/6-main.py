#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
dropout_create_layer = __import__('6-dropout_create_layer').dropout_create_layer

if __name__ == '__main__':
    tf.set_random_seed(0)
    np.random.seed(0)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    X = np.random.randint(0, 256, size=(10, 784))
    a = dropout_create_layer(x, 256, tf.nn.tanh, 0.8)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a, feed_dict={x: X}))

# Expected Output:
# 2018-11-26 21:00:33.541659: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
# [[-1.  1. -1. ...  1. -1. -1.]
#  [-1.  1.  1. ...  1. -1. -1.]
#  [-1. -1. -1. ... -1.  1. -1.]
#  ...
#  [-1. -1. -1. ...  1. -1. -1.]
#  [-1. -1. -1. ...  1. -1. -1.]
#  [-1.  1. -1. ...  1. -1. -1.]]