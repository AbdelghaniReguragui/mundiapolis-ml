
#!/usr/bin/env python3

import tensorflow as tf

def create_placeholders(nx, classes):
    p1 = tf.placeholder("float", shape=[None, nx], name='x')
    p2 = tf.placeholder("float", shape=[None, classes], name='y')
    return p1, p2
