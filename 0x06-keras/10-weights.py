#!/usr/bin/env python3
"""
Save and Load Weights
"""
import tensorflow.keras as K

"""
saves a model’s weights:
network is the model whose weights should be saved
filename is the path of the file that the weights should be saved to
save_format is the format in which the weights should be saved
Returns: None
"""
def save_weights(network, filename, save_format='h5'):
    network.save_weights('./' + filename, save_format=save_format)
    return None

"""
loads a model’s weights:
network is the model to which the weights should be loaded
filename is the path of the file that the weights should be loaded from
Returns: None
"""
def load_weights(network, filename):
    network.load_weights('./' + filename)
    return None
