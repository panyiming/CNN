# encoding: utf-8
# softmax layer


import numpy as np


class Softmax:

    def __init__(self, class_num):
        self._class_num = class_num
        self.update_weight = False
        self._init_params = {'class_num':class_num}
        self._label_eye = np.eye(self._class_num)
    
    def get_params(self):
        params = {'class':'Softmax'}
        params['init_params'] = self._init_params
        return params
    
    def forward(self, in_array):
        n, _ = in_array.shape
        x_exp = np.exp(in_array)
        x_sum = np.sum(x_exp, axis=1).reshape(n, 1)
        x_sum = np.broadcast_to(x_sum, (n, self._class_num))
        out_pro = x_exp / x_sum
        self._out_pro = out_pro
        return out_pro

    def backward(self, labels, lr):
        one_hot_target = self._label_eye[labels]
        out_grad = self._out_pro - one_hot_target
        return out_grad
    

def acc(pred, labels, class_num):
    n = labels.shape[0]
    label_eye = np.eye(class_num)
    label_hot = label_eye[labels]
    a = (label_hot>0)==(pred>0.5)
    right_num = np.sum((label_hot>0)&(pred>0.5))
    acc = right_num / n
    return acc
