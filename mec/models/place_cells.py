# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import scipy


class PlaceCells(object):
    def __init__(self, options):
        self.place_cell_identity = options.place_cell_identity
        self.Np = options.Np
        if self.place_cell_identity:
            assert self.Np == 2
        self.sigma = options.place_cell_rf
        self.surround_scale = options.surround_scale
        self.min_x = options.min_x
        self.max_x = options.max_x
        self.min_y = options.min_y
        self.max_y = options.max_y
        self.DoG = options.DoG

        # Randomly tile place cell centers across environment
        tf.random.set_seed(0)
        usx = tf.random.uniform((self.Np,), self.min_x, self.max_x, dtype=tf.float64)
        usy = tf.random.uniform((self.Np,), self.min_y, self.max_y, dtype=tf.float64)
        self.us = tf.stack([usx, usy], axis=-1)

    def get_activation(self, pos):
        """
        Get place cell activations for a given position.

        Args:
            pos: 2d position of shape [batch_size, sequence_length, 2].

        Returns:
            outputs: Place cell activations with shape [batch_size, sequence_length, Np].
        """
        if self.place_cell_identity:
            outputs = tf.cast(tf.identity(pos), dtype=tf.float32)
        else:
            d = tf.abs(pos[:, :, tf.newaxis, :] - self.us[tf.newaxis, tf.newaxis, ...])

            norm2 = tf.reduce_sum(d ** 2, axis=-1)

            # Normalize place cell outputs with prefactor alpha=1/2/np.pi/self.sigma**2,
            # or, simply normalize with softmax, which yields same normalization on
            # average and seems to speed up training.
            outputs = tf.nn.softmax(-norm2 / (2 * self.sigma ** 2))

            if self.DoG:
                # Again, normalize with prefactor
                # beta=1/2/np.pi/self.sigma**2/self.surround_scale, or use softmax.
                outputs -= tf.nn.softmax(
                    -norm2 / (2 * self.surround_scale * self.sigma ** 2)
                )

                # Shift and scale outputs so that they lie in [0,1].
                outputs += tf.abs(tf.reduce_min(outputs, axis=-1, keepdims=True))
                outputs /= tf.reduce_sum(outputs, axis=-1, keepdims=True)
        return outputs

    def get_nearest_cell_pos(self, activation, k=3):
        """
        Decode position using centers of k maximally active place cells.

        Args:
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].
        """
        if self.place_cell_identity:
            pred_pos = tf.cast(tf.identity(activation), dtype=tf.float32)
        else:
            _, idxs = tf.math.top_k(activation, k=k)
            pred_pos = tf.reduce_mean(tf.gather(self.us, idxs), axis=-2)
        return pred_pos
