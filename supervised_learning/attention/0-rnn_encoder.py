#!/usr/bin/env python3
""" Module Attention """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
        Encode for machine translation.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
            Class constructor
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
            Initialize the hidden states
            for the RNN cell to a tensor of zeros
        """
        initiliazer = tf.keras.initializers.Zeros()
        hiddenQ = initiliazer(shape=(self.batch, self.units))
        return hiddenQ

    def call(self, x, initial):
        """
            Call methods
        """
        embedding = self.embedding(x)
        outputs, hidden = self.gru(embedding, initial_state=initial)
        return outputs, hidden
