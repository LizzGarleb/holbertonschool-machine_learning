#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
learning_rate_decay = __import__('11-learning_rate_decay').learning_rate_decay

if __name__ == '__main__':
    alpha_init = 0.1
    for i in range(100):
        alpha = learning_rate_decay(alpha_init, 1, i, 10)
        print(alpha)

# Expected Output:
# 0.1
# 0.1
# 0.1
# 0.1
# 0.1
# 0.1
# 0.1
# 0.1
# 0.1
# 0.1
# 0.05
# 0.05
# 0.05
# 0.05
# 0.05
# 0.05
# 0.05
# 0.05
# 0.05
# 0.05
# 0.03333333333333333
# 0.03333333333333333
# 0.03333333333333333
# 0.03333333333333333
# 0.03333333333333333
# 0.03333333333333333
# 0.03333333333333333
# 0.03333333333333333
# 0.03333333333333333
# 0.03333333333333333
# 0.025
# 0.025
# 0.025
# 0.025
# 0.025
# 0.025
# 0.025
# 0.025
# 0.025
# 0.025
# 0.02
# 0.02
# 0.02
# 0.02
# 0.02
# 0.02
# 0.02
# 0.02
# 0.02
# 0.02
# 0.016666666666666666
# 0.016666666666666666
# 0.016666666666666666
# 0.016666666666666666
# 0.016666666666666666
# 0.016666666666666666
# 0.016666666666666666
# 0.016666666666666666
# 0.016666666666666666
# 0.016666666666666666
# 0.014285714285714287
# 0.014285714285714287
# 0.014285714285714287
# 0.014285714285714287
# 0.014285714285714287
# 0.014285714285714287
# 0.014285714285714287
# 0.014285714285714287
# 0.014285714285714287
# 0.014285714285714287
# 0.0125
# 0.0125
# 0.0125
# 0.0125
# 0.0125
# 0.0125
# 0.0125
# 0.0125
# 0.0125
# 0.0125
# 0.011111111111111112
# 0.011111111111111112
# 0.011111111111111112
# 0.011111111111111112
# 0.011111111111111112
# 0.011111111111111112
# 0.011111111111111112
# 0.011111111111111112
# 0.011111111111111112
# 0.011111111111111112
# 0.01
# 0.01
# 0.01
# 0.01
# 0.01
# 0.01
# 0.01
# 0.01
# 0.01
# 0.01