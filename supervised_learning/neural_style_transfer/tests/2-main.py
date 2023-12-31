#!/usr/bin/env python3

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import os

NST = __import__('2-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    # Reproducibility
    seed=31415
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    nst = NST(style_image, content_image)
    input_layer = tf.constant(np.random.randn(1, 28, 30, 3), dtype=tf.float32)
    gram_matrix = nst.gram_matrix(input_layer)
    print(gram_matrix)

# Expected Output:
# tf.Tensor(
# [[[ 0.9555436  -0.00985771  0.01343341]
#   [-0.00985771  0.99867874  0.03569118]
#   [ 0.01343341  0.03569118  0.98452365]]], shape=(1, 3, 3), dtype=float32)
