#!/usr/bin/env python3
"""
Evaluate
"""
import tensorflow as tf

def evaluate(X, Y, save_path):
  """
  X is a numpy.ndarray containing the input data to evaluate
  Y is a numpy.ndarray containing the one-hot labels for X
  Returns: the network’s prediction, accuracy, and loss, respectively
  """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        y_eval = sess.run(y_pred, feed_dict={x: X, y: Y})
        accuracy_eval = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss_eval = sess.run(loss, feed_dict={x: X, y: Y})
        return y_eval, accuracy_eval, loss_eval
