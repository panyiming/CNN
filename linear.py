# encoding: utf-8
# linear layer


import numpy as np


class Linear:
    
    def __init__(self, dim, class_num):
        self._dim = dim
        self._class_num = class_num
        self._init_params = {'dim':dim, 'class_num':class_num}
        self.init_weight()

    def init_weight(self, weight=None):
        if weight == None:
            size = (self._dim, self._class_num)
            self._weight = np.random.normal(loc=0.0, scale=0.1, size=size)
        else:
            self._weight = np.array(weight)

    def get_params(self):
        params = {'class':'Linear'}
        params['weight'] = self._weight.tolist()
        params['init_params'] = self._init_params
        return params

    def forward(self, in_array):
        self.in_array = in_array
        out_array = np.dot(in_array, self._weight)
        return out_array

    def _update(self, in_grad, lr):
        w_delta = np.dot(self.in_array.transpose(), in_grad)
        self._weight = self._weight - lr *  self._weight 
        
    def backward(self, in_grad, lr):
        out_grad = np.dot(in_grad, self._weight.transpose())
        self._update(in_grad, lr)
        return out_grad
