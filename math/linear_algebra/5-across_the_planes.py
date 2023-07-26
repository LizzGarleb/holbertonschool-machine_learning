#!/usr/bin/env python3
"""
    Module content:
        - add_matrices2D - Adds two matrices element-wise
        - matrix_shape - Calculates the shape of a matrix
"""
matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(mat1, mat2):
    """
        add_matrices2D - Adds two matrices element-wise

        @mat1 & @mat2: Matrices to be added

        Return: Upon success returns a new matrix with the sum,
                otherwise None
    """
    size1 = matrix_shape(mat1) 
    size2 = matrix_shape(mat2)
    if size1 == size2:
        return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
                for i in range(len(mat1))]
    return None

    # mat3 = []
    # if matrix_shape(mat1) == matrix_shape(mat2):
    #     for row in range(len(mat1)):
    #         rowResult = []
    #         for col in range(len(mat1[row])):
    #             rowResult.append(mat1[row][col] + mat2[row][col])
    #         mat3.append(rowResult)
    #     return mat3
    # return None
