#!/usr/bin/env python3
"""
    Module content:
        - np_transpose: Transpose a matrix
"""


def np_transpose(matrix):
    """
        Compute the transpose of a numpy.ndarray.

        This function takes a numpy array as input and
        calculates its transpose by using the built-in 'T'
        attribute of the array.

        Parameters:
            matrix (numpy.ndarray): The input numpy array
            for which the transpose needs to be computed.

        Returns:
            numpy.ndarray: A new numpy array representing
            he transpose of the input array.
    """
    return matrix.T
