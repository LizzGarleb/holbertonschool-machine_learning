#!/usr/bin/env python3
"""
function def np_slice(matrix, axes={})
"""


def np_slice(matrix, axes={}):
    """slices a matrix along specific axes"""
    # Create a list of slices for each axis
    slices = [slice(None)] * matrix.ndim  # Initialize with slices that select the entire axis
    
    # Update the slices based on the provided axes dictionary
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)
    
    # Use the slices to slice the matrix along the specified axes
    sliced_matrix = matrix[tuple(slices)]
    
    return sliced_matrix
