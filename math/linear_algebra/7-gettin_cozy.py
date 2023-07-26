#!/usr/bin/env python3
"""
    Module content:
        - cat_matrices2D - Concatenates two matrices along
                            a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
        cat_matrices2D - Concatenates two matrices along a
                        specific axis

        @mat1 & @mat2: Matrix to concatenate
        @axis: Options to concatenate the matrix

        Return: a new matrix concatenated

        Description: When axis is 0, mat2 will be added at
        the end of mat1. If axis is 1, concatenation will be
        per column
    """
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    if axis == 1 and len(mat1) != len(mat2):
        return None

    newMatrix = []

    if axis == 0:
        newMatrix = mat1 + mat2

    if axis == 1:
        for i in range(len(mat1)):
            newMatrix.append(mat1[i] + mat2[i])

    return newMatrix
