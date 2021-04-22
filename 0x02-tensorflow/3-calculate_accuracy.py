#!/usr/bin/env python3
import tensorflow as tf

def calculate_accuracy(y, y_pred):
    c_p = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    a = tf.reduce_mean(tf.cast(c_p, tf.float32))
    return a
