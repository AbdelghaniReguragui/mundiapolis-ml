#!/usr/bin/env python3
"""
One hot
"""
import tensorflow.keras as K

def one_hot(labels, classes=None):
    """
    The last dimension of the one-hot matrix must be the number of classes
    Returns: the one-hot matrix
    """
    matrix = K.utils.to_categorical(labels, classes)
    return matrix
