#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    X = X_3D.reshape((X_3D.shape[0], -1))

    tf.set_random_seed(0)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    a = create_batch_norm_layer(x, 256, tf.nn.tanh)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(a, feed_dict={x:X[:5]}))

# Expected Output:
# [[-0.6847082  -0.8220385  -0.35229233 ...  0.464784   -0.8326035
#   -0.96122414]
#  [-0.77318543 -0.66306996  0.7523017  ...  0.811305    0.79587764
#    0.47134086]
#  [-0.21438502 -0.11646973 -0.59783506 ... -0.95093447 -0.67656237
#    0.26563355]
#  [ 0.3159215   0.93362606  0.8738444  ...  0.26363495 -0.320637
#    0.683548  ]
#  [ 0.9421419   0.37344548 -0.8536682  ... -0.06270568  0.85227346
#    0.3293217 ]]