# encoding: utf-8
# test model layers


import numpy as np
from cov import Cov
from pool import Pool
from flatten import Flatten
from linear import Linear
from relu import Relu
from loss import Softmax, acc


def get_layers(inshape, class_num, dim=10):
    layers = []
    cov1 = Cov(5, 0, 1, 6, inshape)
    layers.append(cov1)
    inshape = cov1.get_outshape()
    relu1 = Relu()
    layers.append(relu1)
    pool1 = Pool(2, 2, 0, inshape)
    layers.append(pool1)
    inshape = pool1.get_outshape()
    cov2 = Cov(5, 0, 1, 16, inshape)
    layers.append(cov2)
    inshape = cov2.get_outshape() 
    relu2 = Relu()
    layers.append(relu2)
    pool2 = Pool(2, 2, 0, inshape)
    layers.append(pool2)
    inshape = pool2.get_outshape()
    flatten = Flatten(inshape)
    layers.append(flatten)
    inshape = flatten.get_outshape()
    linear1 = Linear(inshape, dim)
    layers.append(linear1)
    linear2 = Linear(dim, class_num)
    layers.append(linear2)
    loss = Softmax(class_num)
    layers.append(loss)
    return layers
