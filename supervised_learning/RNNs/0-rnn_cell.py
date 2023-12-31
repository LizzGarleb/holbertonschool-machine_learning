#!/usr/bin/env python3
""" Module RNNs """

import numpy as np


class RNNCell:
    """
      Represent a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
          Class constructor
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
          Perform forward propagation for one time step
        """
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)
        h_next = np.tanh((h_x.T @ self.Wh) + self.bh)
        y_pred = (h_next @ self.Wy) + self.by
        z = y_pred
        y = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return h_next, y
