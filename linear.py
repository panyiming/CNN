# encoding: utf-8
# linear layer


import numpy as np


class Linear:
    
    def __init__(self, dim, class_num):
        self._dim = dim
        self._class_num = class_num
        self.update_weight = True
        self._init_params = {'dim':dim, 'class_num':class_num}
        self.init_weight()

    def init_weight(self, weight=None):
        if weight == None:
            size = (self._dim, self._class_num)
            self._weight = np.random.normal(loc=0.0, scale=1.0, size=size) / 10.0
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

    def update(self, in_grad, lr):
        print(in_grad)
        w_delta = np.dot(self.in_array.transpose(), in_grad)
        self._weight -= lr *  self._weight 
        
    def backward(self, in_grad):
        out_grad = np.dot(in_grad, self._weight.transpose())
        return out_grad
