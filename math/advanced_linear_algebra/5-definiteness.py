#!/usr/bin/env python3
"""Definiteness"""

import numpy as np


def definiteness(matrix):
    """calculates the definiteness of a matrix"""
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) == 1:
        return None
    if (matrix.shape[0] != matrix.shape[1]):
        return None
    if not np.all(matrix.T == matrix):
        return None

    w, v = np.linalg.eig(matrix)
    if np.all(w > 0):
        return "Positive definite"
    if np.all(w >= 0):
        return "Positive semi-definite"
    if np.all(w < 0):
        return "Negative definite"
    if np.all(w <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
