#!/usr/bin/env python3

"""
Module contains matrix_shape()
 - matrix_shape: Calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """matrix_shape - Calculates the shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
