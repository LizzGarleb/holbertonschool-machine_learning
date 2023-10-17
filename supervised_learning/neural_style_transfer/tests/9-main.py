#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

NST = __import__('9-neural_style').NST


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
    image, cost = nst.generate_image(iterations=2000, step=100, lr=0.002)
    print("Best cost:", cost)
    plt.imshow(image)
    plt.show()

# Expected Output:
# Cost at iteration 0: 1382606976.0, content 0.0, style 1382606976.0
# Cost at iteration 100: 52111052.0, content 121.86885833740234, style 50892364.0
# Cost at iteration 200: 21114370.0, content 130.403564453125, style 19810334.0
# Cost at iteration 300: 11243212.0, content 130.1966552734375, style 9941246.0
# Cost at iteration 400: 7096659.0, content 127.32190704345703, style 5823440.0
# Cost at iteration 500: 4966846.0, content 122.15582275390625, style 3745287.5
# Cost at iteration 600: 3672262.0, content 115.41342163085938, style 2518127.75
# Cost at iteration 700: 2825318.0, content 107.56407165527344, style 1749677.375
# Cost at iteration 800: 2265213.0, content 99.66951751708984, style 1268517.875
# Cost at iteration 900: 1908019.5, content 92.91487121582031, style 978870.875
# Cost at iteration 1000: 1679912.75, content 87.49122619628906, style 805000.5625
# Cost at iteration 1100: 1541716.0, content 82.94490814208984, style 712266.875
# Cost at iteration 1200: 1425401.0, content 79.44942474365234, style 630906.75
# Cost at iteration 1300: 1352317.25, content 76.84253692626953, style 583891.9375
# Cost at iteration 1400: 1291177.75, content 74.66423797607422, style 544535.375
# Cost at iteration 1500: 1246144.0, content 72.87549591064453, style 517389.09375
# Cost at iteration 1600: 1208923.25, content 71.35684967041016, style 495354.75
# Cost at iteration 1700: 1176405.0, content 70.07740020751953, style 475631.0625
# Cost at iteration 1800: 1151815.5, content 68.9072265625, style 462743.25
# Cost at iteration 1900: 1136183.5, content 67.92913818359375, style 456892.1875
# Cost at iteration 2000: 1116977.5, content 67.18409729003906, style 445136.5625
# Best cost: 1106760.4