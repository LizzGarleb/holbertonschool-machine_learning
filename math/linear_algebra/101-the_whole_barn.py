#!/usr/bin/env python3
"""
Add two matrices
"""


def add_matrices(mat1, mat2):
    # Check if mat1 and mat2 have the same shape
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None  # Return None if matrices have different shapes

    # Initialize an empty result matrix with the same shape
    result_matrix = [[0 for _ in range(len(mat1[0]))] for _ in range(len(mat1))]

    # Add the matrices element-wise
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            result_matrix[i][j] = mat1[i][j] + mat2[i][j]

    return result_matrix
