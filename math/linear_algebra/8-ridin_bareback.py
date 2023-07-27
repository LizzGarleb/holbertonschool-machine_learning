#!/usr/bin/env python3
"""
    Module content:
        - mat_mul: Matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
        mat_mul - Matrix multiplication

        @mat1 & @mat2: Contains matrix to be multiplied
        @colMat1 & @rowMat1: Have the length of the col & rows
                            of the first matrix
        @colMat2 & @rowMat2: Have the length of the col & rows
                            of the second matrix

        Return: Upon success returns a new matrix with the multiply
        values, otherwise None.
    """
    colMat1, rowMat1 = len(mat1[0]), len(mat1)
    colMat2, rowMat2 = len(mat2[0]), len(mat2)
    if colMat1 != rowMat2:
        return None
    newMatrix = []
    for i in range(rowMat1):
        empty_row = []
        for j in range(colMat2):
            empty_row.append(0)
        newMatrix.append(empty_row)
    for i in range(rowMat1):
        for j in range(colMat2):
            for k in range(rowMat2):
                newMatrix[i][j] += mat1[i][k] * mat2[k][j]
    return newMatrix
