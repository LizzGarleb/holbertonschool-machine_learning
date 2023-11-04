#!/usr/bin/env python3
"""Minor"""


def minor(matrix):
    """calculates the minor matrix of a matrix"""
    mat_l = len(matrix)
    range_mat_l = range(len(matrix))

    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([type(mat) == list for mat in matrix]):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    if not all([len(mat) == mat_l for mat in matrix]):
        raise ValueError("matrix must be a non-empty square matrix")
    if mat_l == 1:
        return [[1]]

    minor_values = []
    for row in range_mat_l:
        minor_r = []
        for col in range_mat_l:
            minor_c = [rows[:col] + rows[col + 1:]
                 for rows in (matrix[:row] + matrix[row + 1:])] 
            minor_r.append(minor_c)
        minor_values.append(minor_r)
    return minor_values
