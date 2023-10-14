#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('2-yolo').Yolo

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
    print('Boxes:', boxes)
    print('Box classes:', box_classes)
    print('Box scores:', box_scores)

# Expected Output:
# WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
# Boxes: [[-213.74336488 -485.47886784  305.68206077  531.53467019]
#  [ -62.82223363  -11.37138215  156.45267787   70.19663572]
#  [ 190.62733946    7.65943712  319.201764     43.75737906]
#  ...
#  [ 647.78041714  491.58472667  662.00736941  502.60750466]
#  [ 586.27543101  487.95333873  715.85860922  499.39422783]
#  [ 666.1128673   481.29683099  728.88754319  501.09378706]]
# Box classes: [19 54 29 ... 63 25 46]
# Box scores: [0.7850503  0.67898563 0.81301861 ... 0.8012832  0.61427808 0.64562072]