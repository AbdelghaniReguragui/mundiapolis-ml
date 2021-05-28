#!/usr/bin/env python3
import tensoflow.keras as tk

def save_model(network, filename):
  model.save('saved_model/my_model')
  return None

def load_model(filename):
  loaded_model = tk.models.load_model(filename)
  return loaded_model
