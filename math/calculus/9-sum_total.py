#!/usr/bin/env python3
"""
    Module content:
        - summation_i_squared: Calculate the sum of squares
                                from 1 to n.
"""

def summation_i_squared(n):
    """
        Calculate the sum of squares from 1 to n.

        @n: The stopping condition.

        Return: The integer value of the sum, otherwise
                none is returned.
    """
    if n > 0:
        return int(n * (n + 1) * (2 * n + 1) / 6)
    return None