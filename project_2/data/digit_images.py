import numpy as np

class Digit:

    def __init__ (self, lab, arr):
        self.label = lab
        self.array = arr


    def get_array(self):
        return self.array

    def get_label(self):
        return self.label

    def set_array(self, arr):
        self.array = arr

    def set_label(self,l):
        self.label = l
