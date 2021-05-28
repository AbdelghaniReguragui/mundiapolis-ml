#!/usr/bin/env python3
import tensorflow.keras as tk

def save_model(network, filename):
  network.save(filename)
  return None

def load_model(filename):
  network = tk.models.load_model(filename)
  return network
