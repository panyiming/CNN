# encoding: utf-8
# test model layers


import numpy as np
from cov import Cov
from pool import Pool
from flatten import Flatten
from linear import Linear
from relu import Relu
from loss import Softmax, acc
from optimizer import Network


def get_layers(inshape, class_num, dim=256):
    layers = []
    cov1 = Cov(5, 0, 1, 6, inshape)
    layers.append(cov1)
    inshape = cov1._get_outshape()
    relu1 = Relu(inshape)
    layers.append(relu1)
    pool1 = Pool(2, 2, 0, inshape)
    layers.append(pool1)
    inshape = pool1._get_outshape()
    cov2 = Cov(5, 0, 1, 16, inshape)
    layers.append(cov2)
    inshape = cov2._get_outshape() 
    relu2 = Relu(inshape)
    layers.append(relu2)
    pool2 = Pool(2, 2, 0, inshape)
    layers.append(pool2)
    inshape = pool2._get_outshape()
    flatten = Flatten(inshape)
    layers.append(flatten)
    linear = Linear(dim, class_num)
    layers.append(linear)
    loss = Softmax(class_num)
    layers.append(loss)
    return layers


if __name__ == '__main__':
    inshape = [3, 28, 28]
    class_num = 10
    lr = 0.1
    layers = get_layers(inshape, class_num)
    net = Network(layers, 8)
    x = np.random.rand(8, 3, 28, 28) / 1000
    out = net.forward(x)
    print(out.shape)
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    net.backward(labels, lr)
