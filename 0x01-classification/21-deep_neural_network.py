#!/usr/bin/env python3
import numpy as np


class DeepNeuralNetwork:

    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        weights = {}
        for i in range(len(layers)):
            if layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            if i == 0:
                weights[key_w] = np.random.randn(layers[i], nx)*np.sqrt(2 / nx)
            else:
                weights[key_w] = np.random.randn(layers[i], layers[
                    i-1]) * np.sqrt(2 / layers[i-1])
            weights[key_b] = np.zeros((layers[i], 1))
        self.__weights = weights

    def forward_prop(self, X):
        self.__cache['A0'] = X
        for i in range(self.__L):
            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            key_cache = 'A' + str(i + 1)
            key_cache_last = 'A' + str(i)
            output_Z = np.matmul(self.__weights[key_w], self.__cache[
                key_cache_last]) + self.__weights[key_b]
            output_A = 1 / (1 + np.exp(-output_Z))
            self.__cache[key_cache] = output_A
        return output_A, self.__cache

    def cost(self, Y, A):
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        for i in reversed(range(self.__L)):

            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            key_cache = 'A' + str(i + 1)
            key_cache_dw = 'A' + str(i)

            A = cache[key_cache]
            A_dw = cache[key_cache_dw]
            if i == self.__L - 1:
                dz = A - Y
                W = self.__weights[key_w]
            else:
                da = A * (1 - A)
                dz = np.matmul(W.T, dz)
                dz = dz * da
                W = self.__weights[key_w]
            dw = np.matmul(A_dw, dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights[key_w] = self.__weights[key_w] - alpha * dw.T
            self.__weights[key_b] = self.__weights[key_b] - alpha * db

    @property
    def cache(self):
        return self.__cache

    @property
    def L(self):
        return self.__L

    @property
    def weights(self):
        return self.__weights
