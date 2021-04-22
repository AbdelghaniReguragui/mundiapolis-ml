#!/usr/bin/env python3

import tensorflow as tf

def create_layer(prev, n, activation):
    i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    la = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=i,
        name='layer'
        )
    return la(prev)
