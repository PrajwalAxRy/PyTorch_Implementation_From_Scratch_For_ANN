import numpy as np 

class Model:
    def __init__(self):
        self.layers = []
    def add(self, layer):
        self.layers.append(layer)