#!/usr/bin/env python3
"""Save and Load Model"""
import tensorflow.keras as tk

def save_model(network, filename):
    """
    network is the model to save
    filename is the path of the file that the model should be saved to
    Returns: None
    """
    network.save(filename)


def load_model(filename):
    """
    filename is the path of the file that the model should be loaded from
    Returns: the loaded model
    """
    model = tk.models.load_model(filename)
    return model
