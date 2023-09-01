#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
create_Adam_op = __import__('10-Adam').create_Adam_op

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    Y = lib['Y_train']
    X = X_3D.reshape((X_3D.shape[0], -1))
    Y_oh = one_hot(Y, 10)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./graph.ckpt.meta')
        saver.restore(sess, './graph.ckpt')
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        train_op = create_Adam_op(loss, 0.001, 0.9, 0.99, 1e-8)
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1000):
            if not (i % 100):
                cost = sess.run(loss, feed_dict={x:X, y:Y_oh})
                print('Cost after {} iterations: {}'.format(i, cost))
            sess.run(train_op, feed_dict={x:X, y:Y_oh})
        cost, Y_pred_oh = sess.run((loss, y_pred), feed_dict={x:X, y:Y_oh})
        print('Cost after {} iterations: {}'.format(1000, cost))

    Y_pred = np.argmax(Y_pred_oh, axis=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_3D[i])
        plt.title(str(Y_pred[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Expected Output:
# 2018-11-09 23:37:09.188702: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
# Cost after 0 iterations: 2.8232274055480957
# Cost after 100 iterations: 0.17724855244159698
# Cost after 200 iterations: 0.0870152935385704
# Cost after 300 iterations: 0.03907731547951698
# Cost after 400 iterations: 0.014239841140806675
# Cost after 500 iterations: 0.0048021236434578896
# Cost after 600 iterations: 0.0018489329377189279
# Cost after 700 iterations: 0.000814757077023387
# Cost after 800 iterations: 0.00038969298475421965
# Cost after 900 iterations: 0.00019614089978858829
# Cost after 1000 iterations: 0.00010206626757280901