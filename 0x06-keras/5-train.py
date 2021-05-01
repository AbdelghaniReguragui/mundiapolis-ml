#!/usr/bin/env python3
"""
Train5 
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    validation_data is the data to validate the model with, if not None
    """
    if validation_data:
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle,
                              validation_data=validation_data)
    else:
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle)
    return history
