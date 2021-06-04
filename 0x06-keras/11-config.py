#!/usr/bin/env python3
import tensorflow.keras as K

def save_config(network, filename):
  model_json = network.to_json()
  with open(filename, "w") as json_file:
    json_file.write(model_json)    
  return none

def load_config(filename):
  with open(filename, 'r') as json_file:
    loaded_model_json = json_file.read()
  network = K.models.model_from_json(loaded_model_json)
  return network
