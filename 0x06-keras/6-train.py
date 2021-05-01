#!/usr/bin/env python3
"""
Train 6
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    early_stopping is a boolean that indicates whether early stopping should be used
    early stopping should only be performed if validation_data exists
    early stopping should be based on validation loss
    patience is the patience used for early stopping
    """
    if validation_data:
        if early_stopping:
            call_back = [K.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=patience)]
        else:
            call_back = None
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=call_back)
    else:
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle)
    return history
