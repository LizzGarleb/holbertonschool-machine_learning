#!/usr/bin/env python3
"""
    Module content:
        - np_elementwise - Performs element-wise addition,
                        subtraction, multiplication, and
                        division.
"""


def np_elementwise(mat1, mat2):
    """
        Perform element-wise operations on two numpy.ndarrays.

        This function takes two numpy arrays as input and performs
        element-wise addition, subtraction, multiplication, and
        division between them. It returns a tuple of numpy arrays,
        each representing a different element-wise operation.

        Parameters:
            mat1 (numpy.ndarray): The first input numpy array for
                                element-wise operations.
            mat2 (numpy.ndarray): The second input numpy array for
                                element-wise operations.

        Returns:
            tuple: A tuple of numpy arrays representing element-wise
            addition, subtraction, multiplication, and division results
            (in that order).
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
