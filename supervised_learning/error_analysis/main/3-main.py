#!/usr/bin/env python3

import numpy as np
specificity = __import__('3-specificity').specificity

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(specificity(confusion))

# Expected Output:
# [0.99218958 0.98777131 0.9865429  0.98599078 0.98750582 0.98399789
#  0.98870119 0.98922476 0.98600469 0.98278237]