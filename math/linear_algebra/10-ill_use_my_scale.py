#!/usr/bin/env python3
"""
    Module content:
        - np_shape: Calculates the shape of a numpy.ndarray
"""


def np_shape(matrix):
    """
        Calculate the shape of a numpy.ndarray.

        This function takes a numpy array as input and
        calculates its shape by using the built-in 'shape'
        attribute of the array.

        Parameters:
            matrix (numpy.ndarray): The input numpy array
            for which the shape needs to be calculated.

        Returns:
            tuple: A tuple of integers representing the
            dimensions of the numpy array.
    """
    return matrix.shape
