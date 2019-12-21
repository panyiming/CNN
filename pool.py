# encoding: utf-8
# pooling layer

import math
import numpy as np
from im2col import im2col, col2im


class Pool:
    
    def __init__(self, kw, s, pad, inshape):
        self._kw = kw
        self._s = s
        self._pad = pad
        self._inshape = inshape
        self._init_params = {'kw':kw, 'pad':pad, 's':s, 'inshape':inshape}
        self.get_outshape()
        self._use_col = False
    
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
        if self._use_col:
            out_array = self._forward_col(in_array)
        else:
            out_array = self._forward(in_array)

        return out_array
    
    def backward(self, in_grad, lr):
        if self._use_col:
            out_grad = self._backward_col(in_grad, lr)
        else:
            out_grad = self._backward(in_grad, lr)
        return out_grad

    def _forward_col(self, in_array):
        n = in_array.shape[0]
        in_c, in_h, in_w = self._inshape
        out_c, out_h, out_w = self._outshape
        self._pad_array = self._padding(in_array)
        pad_array_reshape = self._pad_array.reshape(n*in_c, 1, in_h, in_w)
        pad_shape = pad_array_reshape.shape[1:]
        self._pad_array_col = im2col(pad_array_reshape, self._kw, self._kw,
                              self._s, pad_shape, self._outshape)
        max_idx = np.argmax(self._pad_array_col, axis=0)
        out_array = self._pad_array_col[max_idx, range(max_idx.size)]
        out_array = out_array.reshape(out_h, out_w, n, out_c)
        out_array = out_array.transpose(2, 3, 0, 1)
        self._max_idx = max_idx
        self._pad_shape = pad_shape
        return out_array

    def _backward_col(self, in_grad, lr):
        n = in_grad.shape[0]
        in_c, in_h, in_w = self._inshape
        out_c, out_h, out_w = self._outshape
        out_grad_col = np.zeros_like(self._pad_array_col)
        in_grad = in_grad.transpose(2, 3, 0, 1).ravel()
        out_grad_col[self._max_idx, range(self._max_idx.size)] = in_grad
        out_grad = col2im(out_grad_col, n*in_c, self._pad_shape, self._outshape, 
                          self._kw, self._kw, self._pad, self._s)
        out_grad = out_grad.reshape(self._pad_array.shape)
        out_grad = self._reverse_padding(out_grad)
        return out_grad

    def _forward(self, in_array):
        n = in_array.shape[0]
        in_c = self._inshape[0]
        out_c, out_h, out_w = self._outshape
        self._pad_array = self._padding(in_array)
        out_array = np.zeros((n, out_c, out_h, out_w))
        self._max_idx = np.zeros_like(self._pad_array, dtype=np.int32)
        for h in range(out_h):
            for w in range(out_w):
                st_w = self._s * w
                st_h = self._s * h
                ed_w = st_w + self._kw
                ed_h = st_h + self._kw
                sub_array = self._pad_array[:, :, st_h:ed_h, st_w:ed_w]
                max_array = np.max(sub_array, axis=(2, 3))
                max_array = max_array.reshape(n, in_c, 1, 1)
                out_array[:, :, h, w] = max_array.reshape(n, out_c)
                max_array = np.broadcast_to(max_array, sub_array.shape)
                self._max_idx[:, :, st_h:ed_h, st_w:ed_w] = (sub_array == max_array)
        return out_array

    def _backward(self, in_grad, lr):
        n = in_grad.shape[0]
        out_c, out_h, out_w = self._outshape
        out_grad = np.zeros_like(self._pad_array)
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

if __name__ == '__main__':
    inshape = [16, 112, 112]
    pool = Pool(2, 2, 0, inshape)
    outshape = pool.get_outshape()
    print(outshape)
    in_array = np.random.rand(128, 16, 112, 112)
    in_grad = np.random.rand(128, 16, 56, 56)
    import time
    t1 = time.time()
    for i in range(10):
        print(pool._outshape)
        pool.forward(in_array)
    t2 = time.time()
    print((t2-t1)/10)
    t1 = time.time()
    for i in range(10):
        print(pool._outshape)
        pool.backward(in_grad, 0.1)
    t2 = time.time()
    print((t2-t1)/10)
