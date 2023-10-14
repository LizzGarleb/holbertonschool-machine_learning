#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('3-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
    output1 = np.random.randn(13, 13, 3, 85)
    output2 = np.random.randn(26, 26, 3, 85)
    output3 = np.random.randn(52, 52, 3, 85)
    boxes, box_confidences, box_class_probs = yolo.process_outputs([output1, output2, output3], np.array([500, 700]))
    boxes, box_classes, box_scores = yolo.filter_boxes(boxes, box_confidences, box_class_probs)
    boxes, box_classes, box_scores = yolo.non_max_suppression(boxes, box_classes, box_scores)
    print('Boxes:', boxes)
    print('Box classes:', box_classes)
    print('Box scores:', box_scores)

# Expected Output:
# WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
# Boxes: [[ 483.49145347  128.010205    552.78146847  147.87465464]
#  [ -38.91328475  332.66704009  102.94594841  363.78584864]
#  [  64.10861893  329.13266621  111.87941603  358.37523958]
#  ...
#  [ -30.48917643  444.88068667   52.18360562  467.77199113]
#  [-111.73190894  428.19222574  175.04566042  483.40040996]
#  [ 130.0729606   467.20024928  172.42160784  515.90336094]]
# Box classes: [ 0  0  0 ... 79 79 79]
# Box scores: [0.80673525 0.80405611 0.78972362 ... 0.6329012  0.61789273 0.61758194]