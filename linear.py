# encoding: utf-8
# linear layer

             
import numpy as np


class Linear:
    
    def __init__(self, dim, class_num, batch_size):
        self._dim = dim
        self._batch_size = batch_size
        self._class_num = class_num

    def _init_weight(self):
        self._weight = np.random.rand(self._dim, self._class_num)

    def forward(self, in_array):
        self.in_array = in_array
        out_array = np.dot(in_array, self._weight)
        return out_array

    def update(self, in_grad, lr):
        w_delta = np.dot(self.in_array.transpose(), in_grad)
        self._weight -= lr *  self._weight 
        
    def backward(self, in_grad):
        out_grad = np.dot(in_grad, self._weight.transpose())
        return out_grad
