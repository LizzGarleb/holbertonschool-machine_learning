#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    from multinormal import MultiNormal

    np.random.seed(0)
    data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    print(mn.mean)
    print(mn.cov)

# Expected Output:
# [[12.04341828]
#  [29.92870885]
#  [10.00515808]]
# [[ 36.2007391  -29.79405239  15.37992641]
#  [-29.79405239  97.77730626 -20.67970134]
#  [ 15.37992641 -20.67970134  24.93956823]]
