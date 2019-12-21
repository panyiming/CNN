# encoding: utf-8
# network


import os
import json
from cov import Cov
from pool import Pool
from flatten import Flatten
from linear import Linear
from relu import Relu
from loss import Softmax, acc


class Network:

    def __init__(self, layers):
        self.layers = layers
        self._layer_num = len(layers)
    
    def forward(self, in_array):
        for l in self.layers:
            in_array = l.forward(in_array)
        return in_array

    def backward(self, labels, lr):
        in_grad = labels 
        for idx in range(self._layer_num):
            layer_idx = self._layer_num - 1 - idx
            out_grad = self.layers[layer_idx].backward(in_grad, lr)
            in_grad = out_grad


def save(net, epoch, model_name, save_root):
    model_name = '{}-{}.json'.format(model_name, str(epoch).zfill(4))
    file_path = os.path.join(save_root, model_name)
    with open(file_path, 'w') as f:
        for l in net.layers:
            l_params = l.get_params()
            f.write(json.dumps(l_params)+'\n')


def load(path):
    layers = []
    with open(path, 'rb') as f:
        for l in f:
            params = json.loads(l.decode().strip())
            class_type = eval(params['class'])
            if 'init_params' in params:
                layer = class_type(**params['init_params'])
            else:
                layer = class_type()
            if 'weight' in params:
                layer.init_weight(params['weight'])
            layers.append(layer)
    return layers


def init_model(model_path):
    layers = load(model_path)
    net = Network(layers)
    return net
