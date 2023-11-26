#!/usr/bin/env python3

import numpy as np
forward = __import__('3-forward').forward

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
    P, F = forward(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(F)

# Expected Output:
# 1.7080966131859584e-214
# [[0.00000000e+000 0.00000000e+000 2.98125000e-004 ... 0.00000000e+000
#   0.00000000e+000 0.00000000e+000]
#  [2.00000000e-002 0.00000000e+000 3.18000000e-003 ... 0.00000000e+000
#   0.00000000e+000 0.00000000e+000]
#  [2.50000000e-001 3.31250000e-002 0.00000000e+000 ... 2.13885975e-214
#   1.17844112e-214 0.00000000e+000]
#  [1.00000000e-002 4.69000000e-002 0.00000000e+000 ... 2.41642482e-213
#   1.27375484e-213 9.57568349e-215]
#  [0.00000000e+000 8.00000000e-004 0.00000000e+000 ... 1.96973759e-214
#   9.65573676e-215 7.50528264e-215]]