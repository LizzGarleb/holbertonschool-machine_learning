#!/usr/bin/env python3
""" Module RNNs """

import numpy as np


class BidirectionalCell:
    """
      Represent a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
          Class constructor
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
          Calculate the hidden state in the
          forward direction for one time step
        """
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)
        h_next = np.tanh((h_x.T @ self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
          Calculate the hidden state in the
          backward direction for one time step
        """
        h_x = np.concatenate((h_next.T, x_t.T), axis=0)
        h_prev = np.tanh((h_x.T @ self.Whb) + self.bhb)
        return h_prev
