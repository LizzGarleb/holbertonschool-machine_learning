#!/usr/bin/env python3
""" PCA"""
import numpy as np


def pca(X, ndim):
    """
      Perform PCA on a dataset.
    """
    X_mean = X - np.mean(X, axis=0)
    u, Sigma, vh = np.linalg.svd(X_mean)
    W = vh.T
    Wr = W[:, :ndim]
    T = np.dot(X_mean, Wr)
    return T
