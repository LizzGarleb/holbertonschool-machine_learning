#!/usr/bin/env python3

import numpy as np
precision = __import__('2-precision').precision

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(precision(confusion))

# Expected Output:
# [0.93033841 0.91020543 0.87359199 0.87264628 0.88494492 0.83298922
#  0.90050821 0.90648596 0.86364617 0.84503099]