#!/usr/bin/env python3
"""
Add two matrices
"""


import numpy as np

def add_matrices(mat1, mat2):
    # Convert mat1 and mat2 to NumPy arrays
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)

    # Check if mat1 and mat2 have the same shape
    if mat1.shape != mat2.shape:
        return None  # Return None if matrices have different shapes

    return mat1 + mat2
