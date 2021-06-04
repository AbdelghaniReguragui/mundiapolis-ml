#!/usr/bin/env python3
import tensorflow.keras as K

def save_config(network, filename):
  model_json = network.to_json()
  with open("/"+ filename, "w") as json_file:
    json_file.write(model_json)
  return none;

def load_config(filename):
  json_file = open("/"+filename, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  return loaded_model_json;
