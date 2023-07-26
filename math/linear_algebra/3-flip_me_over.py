#!/usr/bin/env python3
"""
    Module content:
        - matrix_transpose - Returns the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
        matrix_transpose - Returns the transpose of a 2D matrix

        @matrix: 2D matrix to transpose

        Return: Return a new matrix upon success
    """
    num_rows = len(matrix)
    num_columns = len(matrix[0])

    transpose_matrix = [[0 for _ in range(num_rows)]
                        for _ in range(num_columns)]

    for row in range(num_rows):
        for col in range(num_columns):
            transpose_matrix[col][row] = matrix[row][col]

    return transpose_matrix
