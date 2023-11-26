#!/usr/bin/env python3

import numpy as np
backward = __import__('5-backward').backward

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    P, B = backward(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(B)

# Expected Output:
# 1.7080966131859631e-214
# [[1.28912952e-215 6.12087935e-212 1.00555701e-211 ... 6.75000000e-005
#   0.00000000e+000 1.00000000e+000]
#  [3.86738856e-214 2.69573528e-212 4.42866330e-212 ... 2.02500000e-003
#   0.00000000e+000 1.00000000e+000]
#  [6.44564760e-214 5.15651808e-213 8.47145100e-213 ... 2.31330000e-002
#   2.70000000e-002 1.00000000e+000]
#  [1.93369428e-214 0.00000000e+000 0.00000000e+000 ... 6.39325000e-002
#   1.15000000e-001 1.00000000e+000]
#  [1.28912952e-215 0.00000000e+000 0.00000000e+000 ... 5.77425000e-002
#   2.19000000e-001 1.00000000e+000]]