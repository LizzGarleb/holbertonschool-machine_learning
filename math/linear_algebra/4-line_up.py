#!/usr/bin/env python3
"""
    Module content:
        - add_array: Adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
        add_array - Adds two arrays element-wise

        @arr1 & @arr2: Arrays that will be added
        @arr3: New array with the sum of arr1 & arr2

        Return: Upon success arr3 is returned, otherwise None
    """

    arr3 = []
    if len(arr1) == len(arr2):
        arr3 = [(arr1[i] + arr2[i]) for i in range(len(arr1))]
        return arr3
    return None
