#!/usr/bin/env python3
""" Train Module """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """ trains a model using mini-batch gradient descent """
    callbacks = []

    def scheduler(epoch):
        return alpha / (1 + decay_rate * epoch)
    
    if validation_data:
        if early_stopping:
            callbacks.append(K.EarlyStopping(patience=patience))
        if learning_rate_decay:
            callbacks.append(K.LearningRateScheduler(scheduler, verbose=1))
        if save_best and filepath:
            callbacks.append(K.ModelCheckpoint(filepath, save_best_only=True))

    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       callbacks=callbacks,
                       validation_data=validation_data, verbose=verbose,
                       shuffle=shuffle)

