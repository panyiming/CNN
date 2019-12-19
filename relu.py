# encoding: utf-8
# relu 


import math
import numpy as np


class Relu:

    def __init__(self, inshape):
        self._neg_idx = None
        self.update_weight = False
        self._init_params = {'inshape':inshape}
        
    def get_params(self):
        params = {'class':'Relu'}
        params['init_params'] = self._init_params
        return params
    
    def forward(self, in_array):
        self._neg_idx = np.where(in_array<0)
        in_array[self._neg_idx] = 0
        return in_array

    def backward(self, in_grad):
        in_grad[self._neg_idx] = 0
        return in_grad
