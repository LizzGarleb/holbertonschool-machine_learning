#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('0-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
    yolo.model.summary()
    print('Class names:', yolo.class_names)
    print('Class threshold:', yolo.class_t)
    print('NMS threshold:', yolo.nms_t)
    print('Anchor boxes:', yolo.anchors)

# Expected output:
# WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# input_1 (InputLayer)            (None, 416, 416, 3)  0                                            
# __________________________________________________________________________________________________
# conv2d (Conv2D)                 (None, 416, 416, 32) 864         input_1[0][0]                    
# __________________________________________________________________________________________________
# batch_normalization (BatchNorma (None, 416, 416, 32) 128         conv2d[0][0]                     
# __________________________________________________________________________________________________

# ...

# reshape (Reshape)               (None, 13, 13, 3, 85 0           conv2d_58[0][0]                  
# __________________________________________________________________________________________________
# reshape_1 (Reshape)             (None, 26, 26, 3, 85 0           conv2d_66[0][0]                  
# __________________________________________________________________________________________________
# reshape_2 (Reshape)             (None, 52, 52, 3, 85 0           conv2d_74[0][0]                  
# ==================================================================================================
# Total params: 62,001,757
# Trainable params: 61,949,149
# Non-trainable params: 52,608
# __________________________________________________________________________________________________
# Class names: ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
# 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
# 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
# frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
# 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
# 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
# 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
# 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
# 'toothbrush']
# Class threshold: 0.6
# NMS threshold: 0.5
# Anchor boxes: [[[116  90]
#   [156 198]
#   [373 326]]

#  [[ 30  61]
#   [ 62  45]
#   [ 59 119]]

#  [[ 10  13]
#   [ 16  30]
#   [ 33  23]]]
