# encoding: utf-8
# optimizer

import numpy as np
from cov import Cov
from pool import Pool
from flatten import Flatten
from linear import Linear
from loss import Softmax, acc

class Network:

    def __init__(self, layers, batch_size):
        self._layers = layers
        self._layer_num = len(layers)
        self.set_batch(batch_size)
    
    def set_batch(self, batch_size):
        for l in self._layers:
            l.set_bs(batch_size)

    def forward(self, in_array):
        for l in self._layers:
            in_array = l.forward(in_array)
        return in_array

    def backward(self, labels, lr):
        in_grad = labels 
        for idx in range(self._layer_num):
            layer_idx = self._layer_num - 1 - idx
            l = self._layers[layer_idx]
            out_grad = l.backward(in_grad)
            if l.update_weight:
                l.update(in_grad, lr)
            in_grad = out_grad


def save(net, epoch, model_name, save_root):
    net = import_module(net)
    net = net.get_net()
    model_name = model_name + str(epoch).zfill(4)+'.json'
    file_path = os.path.join(save_root, model_name)
    idx = 0
    with open(file_path, 'w') as f:
        for l in net.layers:
            l_params = l.get_params()
            f.write(json.dumps(l_params)+'\n')
            idx += 1


def load(path):
    layers = []
    with open(path, 'rb') as f:
        for l in f:
            parms = json.loads(l.decode().strip())
            class_type = eval(params['class'])
            layer = class_type(**params['init_params'])
            if layer.update_weight:
                layer.init_weight(params['weight'])
            layers.append(layer)
    return layers


def init_model(model_path, batch_size):
    layers = load(model_path)
    net = Network(layers, batch_size)
    return net


def train(layers, img_paths, inshape, 
          model_name, save_root, batch_size, 
          epoch, lr, step_epoch):
    net = Network(layers, batch_size)
    dataloader = DataLoader(img_paths, batch_size, inshape)
    for epoch_i in range(epoch):
        dataloader.reset()
        for imgs, labels in dataloader.load_imgs:
            pred = net.forward(imgs)
            acc = acc(pred, labels)
            print(acc)
            net.backward(labels)
        save(net, epoch_i, model_name, save_root)


def train_main(conf):
    net = import_module(conf.net)
    layers =  net.get_layers(conf.inshape, conf.class_num)
    train(layers, conf.img_paths, conf.inshape,
          conf.model_name, conf.save_root, 
          conf.batch_size, conf.epoch, conf.lr, conf.step_epoch)
