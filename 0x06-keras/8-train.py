#!/usr/bin/env python3
import tensorflow.keras as K
"""
save_best is a boolean indicating whether to save the model after each epoch if it is the best
a model is considered the best if its validation loss is the lowest that the model has obtained
filepath is the file path where the model should be saved
"""

def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):

    if filepath:
        call_backs = [K.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                  save_best_only=True)]
    else:
        call_backs = []
    if validation_data:
        if early_stopping or learning_rate_decay:
            if early_stopping:
                call_backs.append(K.callbacks.EarlyStopping(monitor="val_loss", patience=patience))
            if learning_rate_decay:
                def scheduler(epoch):
                    return alpha * 1/(1 + decay_rate * epoch)
                call_backs.append(K.callbacks.LearningRateScheduler(scheduler, verbose=True))
        history = network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle, validation_data=validation_data, callbacks=call_backs)
    else:
        history = network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle, call_backs=call_backs)
    return history
