#!/usr/bin/env python3
"""
    Module Content:
        - A Normal class that represents a normal distribution.
"""


class Normal:

    """
        Functions:
            - __init__: Class constructor
            - z_score(self, x): Calculates the z-score of a given x-value
                Return: the z-score of x
            - x_value(self, z): Calculates the x-value of a given z-score
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
            __init__: class constructor

            @data: list of the data to be used to estimate the distribution
            @mean: is the mean of the distribution
            @stddev: is the standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            sigma = 0
            for i in range(0, len(data)):
                x = (data[i] - self.mean) ** 2
                sigma += x
            self.stddev = (sigma / len(data)) ** (1 / 2)

    def z_score(self, x):
        """
            z_score: Calculates the z-score of a given x-value

            @x: is the x-value

            Return: the z-score of x
        """
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        """
            x_value: Calculates the x-value of a given z-score

            @z: is the z-score

            Return: the x-value of z
        """
        return ((z * self.stddev) + self.mean)
