#!/usr/bin/env python3
"""
    Module Content:
        - A Poisson class that represent a poisson distribution.
"""


class Poisson:

    """
        Functions:
            - __init__(self, data=None, lambtha=1.): Class constructor
            - pmd: Calculate the value of the PMF for a given number of
                    "successes"
                    Returns: the PMF value of k
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
            self.lambtha = float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)