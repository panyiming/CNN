# encoding: utf-8
# optimizer


import os
import json
import numpy as np
from data import DataLoader
from importlib import import_module
from loss import acc
from tqdm import tqdm
from mylog import logger
from network import Network, load, save, init_model


filename = os.path.basename(__file__)
logger = logger(filename)


def train(net, img_paths, inshape, model_name, 
          save_root, batch_size, epoch, lr, 
          step_epoch, class_num, log_step=10):
    dataloader = DataLoader(img_paths, batch_size, inshape)
    step_num = dataloader.step_num
    for epoch_i in range(epoch):
        dataloader.reset()
        step = 0
        if epoch_i in step_epoch:
            lr = lr * 0.1
        for imgs, labels in dataloader.load_imgs():
            step += 1
            pred = net.forward(imgs)
            ac = acc(pred, labels, class_num)
            if step % log_step == 0:
                logger.info('[{}/{}]====[{}/{}]====[lr:{}]====[{}]'.format(
                            epoch_i, epoch, step, step_num, lr, ac))
            net.backward(labels, lr)
        save(net, epoch_i, model_name, save_root)


def test(model_path, img_paths, inshape, batch_size, class_num):
    net = init_model(model_path)
    dataloader = DataLoader(img_paths, batch_size, inshape)
    accs = []
    step_num = dataloader.step_num
    pbar = tqdm(total=step_num)
    for imgs, labels in dataloader.load_imgs():
        pbar.update(1)
        pred = net.forward(imgs)
        ac = acc(pred, labels, class_num)
        accs.append(ac)
    acc_all = np.mean(np.array(accs))
    logger.info('th acc is {}'.format(acc_all))


def train_main(conf):
    if conf.model_path == None:
        model_net = import_module(conf.net)
        layers =  model_net.get_layers(conf.inshape, conf.class_num)
        net = Network(layers)
    else:
        net = init_model(conf.model_path)
    train(net, conf.imgs_path, conf.inshape,
          conf.model_name, conf.save_root, 
          conf.batch_size, conf.epoch, conf.lr,
          conf.step_epoch, conf.class_num)

def test_main(conf):
    test(conf.model_path, conf.imgs_path, 
         conf.inshape, conf.batch_size, conf.class_num)
