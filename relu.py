# encoding: utf-8
# relu 


import math
import numpy as np


class Relu:

    def __init__(self):
        self._neg_idx = None
        
    def get_params(self):
        params = {'class':'Relu'}
        return params
    
    def forward(self, in_array):
        self._neg_idx = np.where(in_array<0)
        in_array[self._neg_idx] = 0
        return in_array

    def backward(self, in_grad, lr):
        in_grad[self._neg_idx] = 0
        return in_grad
