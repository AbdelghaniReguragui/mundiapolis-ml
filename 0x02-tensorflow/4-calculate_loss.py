#!/usr/bin/env python3
import tensorflow as tf

def calculate_loss(y, y_pred):

    lo = tf.losses.softmax_cross_entropy(y, y_pred)
    return lo
