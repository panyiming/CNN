# encoding: utf-8
# pooling layer

import math
import numpy as np


class Pool:
    
    def __init__(self, kw, s, pad, inshape):
        self._kw = kw
        self._s = s
        self._pad = pad
        self._inshape = inshape
        self.update_weight = False
        self._init_params = {'kw':kw, 'pad':pad, 's':s, 'inshape':inshape}

    def set_bs(self, batch_size=1):
        self._batch_size = batch_size
        self._get_outshape()
        self.out_array = np.zeros(self._outshape)
    
    def get_params(self):
        params = {'class':'Pool'}
        params['init_params'] = self._init_params
        return params
    
    def _get_outshape(self):
        c, h, w = inshape
        out_h = (h + 2 * self._pad - self._kw ) / self._s + 1
        out_h = math.floor(out_h)
        out_w = (w + 2 * self._pad - self._kw ) / self._s + 1
        out_w = math.floor(out_w)
        self._outshape = [self._batch_size, c, out_h, out_w]
    
    def _padding(self, array):
        p_w = self._pad
        pad = ((0, 0), (0, 0), (p_w, p_w), (p_w, p_w))
        array = np.pad(array,  pad_width=pad, mode='constant', constant_values=0)
        return array
    
    def _reverse_padding(self, array):
        p_w = self._pad
        n, c, w, h = array.shape
        st_w = p_w
        st_h = p_w
        ed_w = w - p_w
        ed_h = h - p_w
        array = array[:, :, st_w:ed_w, st_h: ed_h]
        return array

    def forward(self, in_array):
        pad_array = self._padding(in_array)
        self.pad_array = pad_array
        n, out_c, out_h, out_w = self.out_array.shape
        for h in range(out_h):
            for w in range(out_w):
                st_w = self._s * w
                st_h = self._s * h
                ed_w = st_w + self._kw
                ed_h = st_h + self._kw
                max_cell = np.max(pad_array[:, :, st_h:ed_h, st_w:ed_w], axis=(2, 3))
                self.out_array[:, :, h, w] = max_cell.reshape(n, out_c)
        return self.out_array

    def backward(self, in_grad):
        out_grad = np.zeros_like(self.pad_array)
        n, out_c, out_h, out_w = in_grad.shape
        for h in range(out_h):
            for w in range(out_w):
                st_w = self._s * w
                st_h = self._s * h
                ed_w = st_w + self._kw
                ed_h = st_h + self._kw
                grad_sub = in_grad[:, :, h, w].reshape(n, out_c, 1, 1)
                grad_sub = np.broadcast_to(grad_sub, (n, out_c, self._kw, self._kw))
                out_grad[:, :, st_h:ed_h, st_w:ed_w] = grad_sub
        out_grad = self._reverse_padding(out_grad)
        return out_grad

if __name__ == '__main__':
    kw = 2
    s = 2
    pad = 1
    inshape = [3, 112, 96]
    batch_size = 8
    in_array = np.random.rand(8, 3, 112, 96)
    pool = Pool(kw, s, pad, inshape)
    pool.set_bs(8)
    out_array = pool.forward(in_array)
    print(out_array.shape)
    in_grad = np.random.rand(8, 3, 57, 49)
    out_grad = pool.backward(in_grad)
    print(out_grad.shape)
