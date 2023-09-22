#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
lenet5 = __import__('4-lenet5').lenet5

if __name__ == "__main__":
    np.random.seed(0)
    tf.set_random_seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    Y_train = lib['Y_train']
    X_valid = lib['X_valid']
    Y_valid = lib['Y_valid']
    m, h, w = X_train.shape
    X_train_c = X_train.reshape((-1, h, w, 1))
    X_valid_c = X_valid.reshape((-1, h, w, 1))
    x = tf.placeholder(tf.float32, (None, h, w, 1))
    y = tf.placeholder(tf.int32, (None,))
    y_oh = tf.one_hot(y, 10)
    y_pred, train_op, loss, acc = lenet5(x, y_oh)
    batch_size = 32
    epochs = 10
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            cost, accuracy = sess.run((loss, acc), feed_dict={x:X_train_c, y:Y_train})
            cost_valid, accuracy_valid = sess.run((loss, acc), feed_dict={x:X_valid_c, y:Y_valid})
            print("After {} epochs: {} cost, {} accuracy, {} validation cost, {} validation accuracy".format(epoch, cost, accuracy, cost_valid, accuracy_valid))
            p = np.random.permutation(m)
            X_shuffle = X_train_c[p]
            Y_shuffle = Y_train[p]
            for i in range(0, m, batch_size):
                X_batch = X_shuffle[i:i+batch_size]
                Y_batch = Y_shuffle[i:i+batch_size]
                sess.run(train_op, feed_dict={x:X_batch, y:Y_batch})
        cost, accuracy = sess.run((loss, acc), feed_dict={x:X_train_c, y:Y_train})
        cost_valid, accuracy_valid = sess.run((loss, acc), feed_dict={x:X_valid_c, y:Y_valid})
        print("After {} epochs: {} cost, {} accuracy, {} validation cost, {} validation accuracy".format(epochs, cost, accuracy, cost_valid, accuracy_valid))
        Y_pred = sess.run(y_pred, feed_dict={x:X_valid_c, y:Y_valid})
        print(Y_pred[0])
        Y_pred = np.argmax(Y_pred, 1)
        plt.imshow(X_valid[0])
        plt.title(str(Y_valid[0]) + ' : ' + str(Y_pred[0]))
        plt.show()

# Expected Output:
# 2018-12-11 01:13:48.838837: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
# After 0 epochs: 2.2976269721984863 cost, 0.08017999678850174 accuracy, 2.2957489490509033 validation cost, 0.08389999717473984 validation accuracy
# After 1 epochs: 0.06289318203926086 cost, 0.9816200137138367 accuracy, 0.0687578096985817 validation cost, 0.9805999994277954 validation accuracy
# After 2 epochs: 0.04042838513851166 cost, 0.987559974193573 accuracy, 0.04974357411265373 validation cost, 0.9861000180244446 validation accuracy
# After 3 epochs: 0.033414799720048904 cost, 0.989300012588501 accuracy, 0.048249948769807816 validation cost, 0.9868000149726868 validation accuracy
# After 4 epochs: 0.03417244181036949 cost, 0.989080011844635 accuracy, 0.06006946414709091 validation cost, 0.983299970626831 validation accuracy
# After 5 epochs: 0.019328827038407326 cost, 0.9940000176429749 accuracy, 0.03986175358295441 validation cost, 0.9883999824523926 validation accuracy
# [7.69712392e-14 1.46036297e-12 1.26758201e-10 9.99998450e-01
#  2.11756339e-14 2.26456431e-09 1.75634965e-13 1.45111270e-10
#  1.56041858e-06 1.31521265e-08]