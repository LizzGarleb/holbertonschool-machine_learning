#!/usr/bin/env python3
"""
    Module Content:
        - A Exponential class that represent an exponential distribution.
"""

class Exponential:

    """
        Functions:
            - def __init__(self, data=None, lambtha=1.): Class constructor

    """

    def __init__(self, data=None, lambtha=1.):
        """
            ___init__: class constructor

            @data: list of data to be used to estimate the distribution
            @lambtha: expected number of occurences in a given time frame

            Raises:
                ValueError: lambtha must be a positive value
                TypeError: data must be a list
                ValueError: data must contain multiple values
        """

        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)