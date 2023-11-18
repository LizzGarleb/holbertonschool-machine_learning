#!/usr/bin/env python3

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    for k in range(1, 11):
        C, _ = kmeans(X, k)
        print('Variance with {} clusters: {}'.format(k, variance(X, C).round(5)))

# Expected Output:
# Variance with 1 clusters: 157927.7052
# Variance with 2 clusters: 82095.68297
# Variance with 3 clusters: 34784.23723
# Variance with 4 clusters: 23158.40095
# Variance with 5 clusters: 7868.521233
# Variance with 6 clusters: 7406.930773
# Variance with 7 clusters: 6930.663613
# Variance with 8 clusters: 6162.158842
# Variance with 9 clusters: 5843.92455
# Variance with 10 clusters: 5727.41124