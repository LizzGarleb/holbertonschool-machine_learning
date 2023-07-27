#!/usr/bin/env python3
"""
    Module content:
        - np_cat: Concatenates two matrices along a
        specific axis
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
        Concatenate two numpy ndarrays along a specific axis.

        This function takes two numpy ndarrays, `mat1` and `mat2`,
        as input and concatenates them along the specified axis
        using numpy's 'concatenate' function. The resulting array
        will have dimensions determined by the concatenation axis.

        Parameters:
            mat1 (numpy.ndarray): The first input numpy array for
                                concatenation.
            mat2 (numpy.ndarray): The second input numpy array for
                                concatenation.
            axis (int, optional): The axis along which to concatenate
                                the matrices. Default is 0.

        Returns:
            numpy.ndarray: A new numpy array representing the
            concatenated matrix.
    """
    return np.concatenate((mat1, mat2), axis=axis)
