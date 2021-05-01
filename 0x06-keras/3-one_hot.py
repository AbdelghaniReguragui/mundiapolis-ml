#!/usr/bin/env python3

import tensorflow.keras as K

def one_hot(labels, classes=None):

    matrix = K.utils.to_categorical(labels, classes)
    return matrix
