#!/usr/bin/env python3
import tensoflow.keras as tk

def save_model(network, filename):
  model.save(filename)
  return None

def load_model(filename):
  network = tk.models.load_model(filename)
  return network
