#!/usr/bin/env python3
"""
    Module content:
        - poly_derivative: Calculate the derivative of a
        polynomial
"""


def poly_derivative(poly):
    """
        Calculate the derivative of a polynomial.

        @poly: List of coefficients representing a polynomial
    """
    if not isinstance(poly, list) or len(poly) < 1:
        return None

    derivative = []
    for i in range(len(poly) - 1):
        derivative.append(poly[i] * (i + 1))

    if len(derivative) == 1 and derivative[0] == 0:
        return [0]

    return derivative
