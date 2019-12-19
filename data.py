# encoing: utf-8
# data load code


import cv2
import random
import math
import numpy as np


class DataLoader:
    
    def __init__(self, path, batch_size, inshape):
        self._path = path
        self._inshape = inshape
        self._batch_size = batch_size
        self._b_st = 0
        self._parse_imgs_label()

    def _parse_imgs_label(self):
        self._path_labels = []
        self._idx = []
        imgnum = 0
        with open(self._path) as f:
            for l in f:
                idx, label, path = l.strip().split('\t')
                label = int(label)
                self._path_labels.append([path, label])
                imgnum += 1
        self.step_num = math.floor(imgnum / self._batch_size) + 1
        self._imgnum = imgnum

    def reset(self):
        random.shuffle(self._path_labels)
        self._b_st = 0

    def _read_imgs(self, path):
        c, in_h, in_w = self._inshape
        im = cv2.imread(path)
        im = cv2.resize(im, (in_w, in_h))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def _transform(self, im_array, mean=[0.5, 0.5, 0.5], 
                   std=[0.5, 0.5, 0.5]):
        im_array = np.array(im_array)
        im_array = ((im_array / 255.0) - mean) / std
        im_array = np.transpose(im_array, (0, 3, 1, 2))
        return im_array

    def load_imgs(self):
        b_st = self._b_st
        while b_st < self._imgnum:
            imgs = []
            labels = []
            b_ed = min(b_st + self._batch_size, self._imgnum)
            for path_label in self._path_labels[b_st:b_ed]:
                path, label = path_label
                im = self._read_imgs(path)
                imgs.append(im)
                labels.append(label)
            labels = np.asarray(labels)
            imgs = self._transform(imgs)
            b_st = b_ed
            yield imgs, labels
