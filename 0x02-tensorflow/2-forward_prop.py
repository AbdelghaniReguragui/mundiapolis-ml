#!/usr/bin/env python3
"""forward propagation"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    x: is the placeholder for the input data
    layer_sizes" is a list containing the number of nodes in each layer of the network
    activations: is a list containing the activation functions for each layer of the network
    Returns: the prediction of the network in tensor form
    Returns: the prediction of the network in tensor form
    """
    if len(layer_sizes) and len(activations):
        if len(layer_sizes) == len(activations):
            A = x
            for i in range(len(layer_sizes)):
                A = create_layer(A, layer_sizes[i], activations[i])
                
            return A
        
    return None
