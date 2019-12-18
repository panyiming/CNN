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

    def set_bs(self, batch_size=1):
        self._batch_size = batch_size

    def init_weight(self, weight=None):
        if weight == None:
            self._weight = np.random.rand(self._dim, self._class_num)
        else:
            self._weight = np.array(weight)

    def get_params(self):
        params = {'class':'Linear'}
        params['weight'] = self._weight.tolist()
        params['init_params'] = self._init_params
        return params

    def forward(self, in_array):
        self.in_array = in_array
        print(in_array.shape)
        out_array = np.dot(in_array, self._weight)
        return out_array

    def update(self, in_grad, lr):
        w_delta = np.dot(self.in_array.transpose(), in_grad)
        self._weight -= lr *  self._weight 
        
    def backward(self, in_grad):
        out_grad = np.dot(in_grad, self._weight.transpose())
        return out_grad
