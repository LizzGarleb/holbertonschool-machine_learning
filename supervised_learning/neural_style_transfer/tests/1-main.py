#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

NST = __import__('1-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    nst = NST(style_image, content_image)
    nst.model.summary()

# Expected Output:
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         [(None, None, None, 3)]   0         
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, None, None, 64)    1792      
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, None, None, 64)    36928     
# _________________________________________________________________
# block1_pool (AveragePooling2 (None, None, None, 64)    0         
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, None, None, 128)   73856     
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, None, None, 128)   147584    
# _________________________________________________________________
# block2_pool (AveragePooling2 (None, None, None, 128)   0         
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, None, None, 256)   295168    
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, None, None, 256)   590080    
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, None, None, 256)   590080    
# _________________________________________________________________
# block3_conv4 (Conv2D)        (None, None, None, 256)   590080    
# _________________________________________________________________
# block3_pool (AveragePooling2 (None, None, None, 256)   0         
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, None, None, 512)   1180160   
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block4_conv4 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block4_pool (AveragePooling2 (None, None, None, 512)   0         
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, None, None, 512)   2359808   
# =================================================================
# Total params: 15,304,768
# Trainable params: 0
# Non-trainable params: 15,304,768
# _________________________________________________________________
