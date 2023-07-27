#!/usr/bin/env python3
"""
    Module content:
        - np_matmul: Matrix multiplication
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
        Perform matrix multiplication using numpy's matmul.

        This function takes two numpy ndarrays, `mat1` & `mat2`,
        as input and performs matrix multiplication using numpy's
        'matmul' function. The function returns a new numpy ndarray
        that represents the result of the matrix multiplication.

        Parameters:
            mat1 (numpy.ndarray): The first input numpy array for
                                matrix multiplication.
            mat2 (numpy.ndarray): The second input numpy array for
                                matrix multiplication.

        Returns:
            numpy.ndarray: A new numpy array representing the result
            of the matrix multiplication.
    """
    return np.matmul(mat1, mat2)
