#!/usr/bin/env python3

import numpy as np
f1_score = __import__('4-f1_score').f1_score

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(f1_score(confusion))

# Expected Output:
# [0.94161242 0.93802288 0.8580209  0.85856574 0.88884336 0.81917654
#  0.91526771 0.90560928 0.8447821  0.84613074]
