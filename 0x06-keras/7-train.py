#!/usr/bin/env python3
"""
Train 7
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    learning_rate_decay is a boolean that indicates whether learning rate decay should be used
    learning rate decay should only be performed if validation_data exists
    the decay should be performed using inverse time decay
    the learning rate should decay in a stepwise fashion after each epoch
    each time the learning rate updates, Keras should print a message
    alpha is the initial learning rate
    decay_rate is the decay rate
    """
    if validation_data:
        if early_stopping or learning_rate_decay:
            call_backs = []
            if early_stopping:
                call_backs.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                                            patience=patience))
            if learning_rate_decay:
                def scheduler(epoch):
                    return alpha * 1/(1 + decay_rate * epoch)
                call_backs.append(
                            K.callbacks.LearningRateScheduler(scheduler,
                                                              verbose=True))
        else:
            call_backs = None
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=call_backs)
    else:
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle)
    return history
