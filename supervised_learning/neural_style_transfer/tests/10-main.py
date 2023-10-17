#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

NST = __import__('10-neural_style').NST


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
    generated_image, cost = nst.generate_image(iterations=2000, step=100, lr=0.002)
    print("Best cost:", cost)
    plt.imshow(generated_image)
    plt.show()
    mpimg.imsave("starry_gate2.jpg", generated_image)

# Expected Output:
# Cost at iteration 0: 1382946432.0, content 0.0, style 1382606976.0, var 33940.97265625
# Cost at iteration 100: 52768028.0, content 121.8442611694336, style 50916476.0, var 63310.609375
# Cost at iteration 200: 21865732.0, content 130.36721801757812, style 19845140.0, var 71691.984375
# Cost at iteration 300: 12060261.0, content 130.16085815429688, style 9980248.0, var 77840.40625
# Cost at iteration 400: 7952183.0, content 127.24192810058594, style 5854871.5, var 82489.1875
# Cost at iteration 500: 5850672.0, content 122.05924224853516, style 3773796.5, var 85628.3125
# Cost at iteration 600: 4568866.0, content 115.25521087646484, style 2542959.5, var 87335.4453125
# Cost at iteration 700: 3720829.5, content 107.35162353515625, style 1770479.25, var 87683.40625
# Cost at iteration 800: 3152189.0, content 99.6286392211914, style 1288950.625, var 86695.203125
# Cost at iteration 900: 2780311.75, content 93.01751708984375, style 1002611.3125, var 84752.53125
# Cost at iteration 1000: 2530589.0, content 87.8449478149414, style 827339.875, var 82479.96875
# Cost at iteration 1100: 2370930.25, content 83.69252014160156, style 730134.1875, var 80387.09375
# Cost at iteration 1200: 2248433.75, content 80.36737060546875, style 659370.5625, var 78538.953125
# Cost at iteration 1300: 2159631.0, content 77.78072357177734, style 612315.25, var 76950.859375
# Cost at iteration 1400: 2093393.75, content 75.78543853759766, style 579579.6875, var 75595.9765625
# Cost at iteration 1500: 2043047.75, content 74.09213256835938, style 558025.625, var 74410.078125
# Cost at iteration 1600: 1996620.0, content 72.79649353027344, style 535056.1875, var 73359.890625
# Cost at iteration 1700: 1955594.5, content 71.68930053710938, style 514480.8125, var 72422.0703125
# Cost at iteration 1800: 1925553.25, content 70.79610443115234, style 501169.65625, var 71642.2421875
# Cost at iteration 1900: 1895888.5, content 69.93038940429688, style 487452.84375, var 70913.171875
# Cost at iteration 2000: 1871497.75, content 69.1607666015625, style 477484.5, var 70240.5546875
# Best cost: 1871497.8