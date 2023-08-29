#!/usr/bin/env python3

import numpy as np


def normalize(X, m, s):
    """ normalizes (standardizes) a matrix 
        Return: The normalized X matrix
    """
    return ((X - m) / s)
