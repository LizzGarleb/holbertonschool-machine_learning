#!/usr/bin/env python3
"""
    Module Content:
        - A Neuron class that defines a single neuron
        performing binary classification
"""
import numpy as np


class NeuralNetwork:
    """
        NeuralNetwork with one hidden layer performing binary
        classification
    """
    def __init__(self, nx, nodes):
        """
            __init__: class constructor

            Input:
                @nx: the number of input features
                @nodes: the number of nodes found in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be a integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
            Getter function

            Return: the weight vector for the hidden layer
        """
        return self.__W1

    @property
    def W2(self):
        """
            Getter function

            Return: The weight vector for the output neuron
        """
        return self.__W2

    @property
    def b1(self):
        """
            Getter function

            Return: the bias for the hidden layer
        """
        return self.__b1

    @property
    def b2(self):
        """
            Getter function

            Return: the bias for the output neuron
        """
        return self.__b2

    @property
    def A1(self):
        """
            Getter function

            Return: the activated output for the hidden layer
        """
        return self.__A1

    @property
    def A2(self):
        """
            Getter function

            Return: the activated output fot the output neuron
        """
        return self.__A2

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network
        """
        u1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-u1))
        self.__A2 = 1 / (1 + np.exp(-(np.matmul(self.__W2, self.__A1)
                                      + self.__b2)))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                 np.multiply(1-Y, np.log(1.0000001-A)))
        return cost

    def evaluate(self, X, Y):
        """
            Evaluate the neuron's predictions
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        pred = np.where(self.__A2 >= 0.5, 1, 0)
        return pred, cost
