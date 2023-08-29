#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf


def normalization_constants(X):
    """
        Calculates the normalization
        (standardization) constants of a matrix

        Return: the mean and standard deviation of
        each feature, respectively
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
