#!/usr/bin/env python3
"""
    Module Content:
        - A Normal class that represents a normal distribution.
"""


class Normal:

    """
        Functions:
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
            __init__: class constructor

            @data: list of the data to be used to estimate the distribution
            @mean: is the mean of the distribution
            @stddev: is the standard deviation of the distribution
        """
        self.stddev = stddev
        if stddev <= 0:
           raise ValueError("stddev must be a positive value") 

        if data is None:
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            sum = 0
            sum2 = 0
            for i in range(0, len(data)):
                sum += data[i]
                sum2 += ((data[i] - mean) ** 2)
            var = ((sum2 / len(data)) ** 0.5)
            self.stddev = float(var)
            mean = sum / len(data)
        self.mean = float(mean)
