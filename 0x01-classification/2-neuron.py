#!/usr/bin/env python3
import numpy as np

'''
La classe Neuron 
'''
class Neuron:
    '''
    Le constructeur de la classe Neuron
    '''
    def __init__(self, nx):
        '''
        Verification si nx est un entier
        '''
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        '''
        Verification si nx est plus grand de 1.
        '''
        if nx < 1:
            raise ValueError("nx must be a positive integer")
            
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
    
    '''
    La fonction forward_prop 
    '''
    def forward_prop(self, X):
        preactivation = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-preactivation))
        return self.__A
    
''' 
Setters
'''
    @property
    def W(self):
        return self.__W
    
    @property
    def b(self):
        return self.__b
    
    @property
    def A(self):
        return self.__A
