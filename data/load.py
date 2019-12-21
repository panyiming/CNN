#!/usr/bin/env python
# coding=utf-8
import os
import struct
import numpy as np
from PIL import Image

def load_mnist_image(path, filename, type = 'train'):
    full_name = os.path.join(path, filename)
    fp = open(full_name, 'rb')
    buf = fp.read()
    index = 0;
    magic, num, rows, cols = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    for image in range(0, num):
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        im = np.array(im, dtype = 'uint8')
        im = im.reshape(28, 28)
        im = Image.fromarray(im)
        if (type == 'train'):
            isExists = os.path.exists('./train')
            if not isExists:
                os.mkdir('./train')
            im.save('./train/train_%s.bmp' %image, 'bmp')
        if (type == 'test'):
            isExists = os.path.exists('./test')
            if not isExists:
                os.mkdir('./test')
            im.save('./test/test_%s.bmp' %image, 'bmp')

def load_mnist_label(path, filename, type = 'train'):
    full_name = os.path.join(path, filename)
    fp = open(full_name, 'rb')
    buf = fp.read()
    index = 0;
    magic, num = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    Labels = np.zeros(num)

    for i in range(num):
        Labels[i] = np.array(struct.unpack_from('>B', buf, index))
        index += struct.calcsize('>B')

    if (type == 'train'):
        np.savetxt('./train_labels.csv', Labels, fmt='%i', delimiter=',')
    if (type == 'test'):
        np.savetxt('./test_labels.csv', Labels, fmt='%i', delimiter=',')

    return Labels

def get_path_label(im_root, path, lst_name, dataname):
    line_num = 0
    fw = open(lst_name, 'w')
    with open(path, 'rb') as f:
        for l in f:
            parts = l.decode().strip().split(',')
            label = parts[0]
            im_name = '{}_{}.bmp'.format(dataname, line_num)
            im_path = os.path.join(im_root, im_name)
            im_path = os.path.abspath(im_path)
            newl = str(line_num)+ '\t' + label + '\t' + im_path  + '\n'
            line_num += 1
            fw.write(newl)
    fw.close()

            

if __name__ == '__main__':
    path = './'
    test_images = 't10k-images-idx3-ubyte'
    load_mnist_image(path, test_images, 'test')
    test_labels = 't10k-labels-idx1-ubyte'
    load_mnist_label(path, test_labels, 'test')
    im_root = './test'
    label_file = './test_labels.csv'
    lst_name = './test.ls'
    get_path_label(im_root, label_file, lst_name, 'test')

    train_images = 'train-images-idx3-ubyte'
    load_mnist_image(path, train_images, 'train')
    train_labels = 'train-labels-idx1-ubyte'
    load_mnist_label(path, train_labels, 'train')
    im_root = './train'
    label_file = './train_labels.csv'
    lst_name = './train.ls'
    get_path_label(im_root, label_file, lst_name, 'train')
