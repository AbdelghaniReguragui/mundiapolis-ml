
#!/usr/bin/env python3
import tensorflow.keras as tk

def save_model(network, filename):
    network.save(filename)


def load_model(filename):
    model = tk.models.load_model(filename)
    return model
