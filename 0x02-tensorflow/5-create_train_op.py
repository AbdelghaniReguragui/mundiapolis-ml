#!/usr/bin/env python3
import tensorflow as tf


def create_train_op(loss, alpha):

    tmp = tf.train.GradientDescentOptimizer(alpha)
    a = op.minimize(loss)
    return a
