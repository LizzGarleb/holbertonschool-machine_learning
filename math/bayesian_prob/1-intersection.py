#!/usr/bin/env python3
"""Intersection"""

import numpy as np


def intersection(x, n, P, Pr):
    """
      Calculate the intersection of obtaining this data.
    """
    if not isinstance(n, int) or (n <= 0):
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or (x < 0):
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not isinstance(Pr, np.ndarray) or (P.shape != Pr.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    sum = np.sum(Pr)
    if not np.isclose(sum, 1):
        raise ValueError("Pr must sum to 1")
    numerator = np.math.factorial(n)
    denominator = (np.math.factorial(x) * (np.math.factorial(n - x)))
    factorial = numerator / denominator
    P_likelihood = factorial * (np.power(P, x)) * (np.power((1 - P), (n - x)))
    intersection = P_likelihood * Pr
    return intersection
