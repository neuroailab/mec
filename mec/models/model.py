# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.layers import RNN as RNN_wrapper
from tensorflow.keras.models import Model


def pos_loss(x, y):
    return (x - y) ** 2


class UGRNNCell(Layer):
    """Collins et al. 2017.
    Adapted from: https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1622-L1713"""

    def __init__(self, units, activation="tanh", **kwargs):
        self.units = units
        self.state_size = units
        self.activation = activation
        super(UGRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        """Initializations taken from here: https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/rnn/python/ops/core_rnn_cell.py#L126-L188"""
        self.weight = self.add_weight(
            shape=(input_shape[-1] + self.units, input_shape[-1] + self.units),
            initializer="glorot_uniform",
            name="weight",
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(input_shape[-1] + self.units,),
            initializer="zeros",
            name="bias",
            trainable=True,
        )
        self.built = True

    def call(self, inputs, states):
        prev_state = states[0]
        assert prev_state.get_shape().as_list()[-1] == self.units  # consistency
        input_dim = inputs.get_shape().as_list()[-1]
        assert (
            input_dim == self.units
        )  # otherwise elementwise multiply of g * prev_state in new_state update will fail
        cell_inputs = tf.concat([inputs, prev_state], axis=-1)
        rnn_matrix = tf.matmul(cell_inputs, self.weight) + self.bias
        [g_act, c_act] = tf.split(
            axis=-1, num_or_size_splits=[input_dim, self.units], value=rnn_matrix
        )
        c = getattr(tf.keras.activations, self.activation)(c_act)
        g = tf.nn.sigmoid(g_act + 1.0)
        new_state = g * prev_state + (1.0 - g) * c
        new_output = new_state
        return new_output, [new_state]


class RNNBase(Model):
    def __init__(self, options, place_cells):
        super(RNNBase, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.place_cell_identity = options.place_cell_identity
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        self.encoder = Dense(self.Ng, name="encoder")
        self.M = Dense(self.Ng, name="M")
        self.RNN = None
        self.dense = Dense(self.Ng, name="dense", activation=options.activation)
        self.decoder = Dense(self.Np, name="decoder")

        # Loss function
        if self.place_cell_identity:
            assert self.Np == 2
            self.loss_fun = pos_loss
        else:
            self.loss_fun = tf.nn.softmax_cross_entropy_with_logits

    def pre_g(self, inputs):
        """Compute rnn cell activations"""
        assert self.RNN is not None
        v, p0 = inputs
        s0 = self.encoder(p0)
        init_state = s0
        Mv = self.M(v)
        rnn = self.RNN(Mv, initial_state=init_state)
        return rnn

    def g(self, inputs):
        """Compute grid cell activations"""
        rnn = self.pre_g(inputs)
        g = self.dense(rnn)
        return g

    def dc(self, inputs):
        g = self.g(inputs)
        return self.decoder(g)

    def call(self, inputs):
        """Predict place cell code"""
        place_preds = self.dc(inputs)

        return place_preds

    def compute_loss(self, inputs, pc_outputs, pos):
        """Compute loss and decoding error"""
        preds = self.call(inputs)
        loss = tf.reduce_mean(self.loss_fun(pc_outputs, preds))

        # # Weight regularization
        loss += self.weight_decay * tf.reduce_sum(self.RNN.weights[1] ** 2)

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = tf.reduce_mean(
            tf.sqrt(
                tf.reduce_sum(
                    (tf.cast(pos, dtype=pred_pos.dtype) - pred_pos) ** 2, axis=-1
                )
            )
        )

        return loss, err


class UGRNN(RNNBase):
    def __init__(self, options, place_cells):
        super(UGRNN, self).__init__(options=options, place_cells=place_cells)
        self.RNN = RNN_wrapper(
            UGRNNCell(self.Ng, activation=options.activation), return_sequences=True
        )


class CueRNNBase(RNNBase):
    """
    Gets cues as additional input.
    Same as RNNBase, but encodes the extra cue input (c) jointly with the velocity input.
    """

    def __init__(self, **kwargs):
        super(CueRNNBase, self).__init__(**kwargs)

    def pre_g(self, inputs):
        assert self.RNN is not None
        v, p0, c = inputs
        s0 = self.encoder(p0)
        init_state = s0
        Mv = self.M(tf.concat([v, c], axis=-1))
        rnn = self.RNN(Mv, initial_state=init_state)
        return rnn


class CueUGRNN(CueRNNBase):
    def __init__(self, options, place_cells):
        super(CueUGRNN, self).__init__(options=options, place_cells=place_cells)
        self.RNN = RNN_wrapper(
            UGRNNCell(self.Ng, activation=options.activation), return_sequences=True
        )
