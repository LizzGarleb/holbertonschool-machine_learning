#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf


def normalize(X, m, s):
    """ normalizes (standardizes) a matrix 
        Return: The normalized X matrix
    """
    return ((X - m) / s)
