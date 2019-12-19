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
        self._init_params = {'kw':kw, 'pad':pad, 's':s, 'inshape':inshape}
        self.get_outshape()
    
    def get_params(self):
        params = {'class':'Pool'}
        params['init_params'] = self._init_params
        return params
    
    def get_outshape(self):
        c, h, w = self._inshape
        out_h = (h + 2 * self._pad - self._kw ) / self._s + 1
        out_h = math.floor(out_h)
        out_w = (w + 2 * self._pad - self._kw ) / self._s + 1
        out_w = math.floor(out_w)
        self._outshape = [c, out_h, out_w]
        return self._outshape
    
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
        n, in_c, _, _  = in_array.shape
        out_c, out_h, out_w = self._outshape
        pad_array = self._padding(in_array)
        self.pad_array = pad_array
        out_array = np.zeros((n, out_c, out_h, out_w))
        self._max_idx = np.zeros_like(pad_array, dtype=np.int32)
        for h in range(out_h):
            for w in range(out_w):
                st_w = self._s * w
                st_h = self._s * h
                ed_w = st_w + self._kw
                ed_h = st_h + self._kw
                sub_array = pad_array[:, :, st_h:ed_h, st_w:ed_w]
                max_array = np.max(pad_array[:, :, st_h:ed_h, st_w:ed_w], axis=(2, 3))
                out_array[:, :, h, w] = max_array.reshape(n, out_c)
                sub_shape = sub_array.shape
                max_array_reshape = np.broadcast_to(max_array.reshape(n, in_c, 1, 1), sub_shape)
                self._max_idx[:, :, st_h:ed_h, st_w:ed_w] = (sub_array == max_array_reshape)
        return out_array

    def backward(self, in_grad, lr):
        print(np.max(in_grad))
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
        out_grad = self._max_idx * out_grad
        out_grad = self._reverse_padding(out_grad)
        return out_grad
