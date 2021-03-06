#!/usr/bin/env python3
import numpy as np

class Neuron:
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
    # W getter
    @property
    def W(self):
        return self.__W

    # b getter
    @property
    def b(self):
        return self.__b

    # A getter
    @property
    def A(self):
        return self.__A
