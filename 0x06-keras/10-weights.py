#!/usr/bin/env python3
import tensorflow.keras as tk

def save_weights('./' +network, filename, save_format='h5'):
  network.save_weights(filepath, save_format=save_format)
  return None

def load_weights(network, filename):
  network.load_weights('./' +filepath)
  return None
