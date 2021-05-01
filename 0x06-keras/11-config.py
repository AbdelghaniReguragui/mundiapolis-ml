#!/usr/bin/env python3
import tensorflow.keras as K


def save_config(network, filename):
    json_conf = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(json_conf)

def load_config(filename):
    with open(filename, "r") as json_file:
        json_conf = json_file.read()
    network = K.models.model_from_json(json_conf)
    return network
