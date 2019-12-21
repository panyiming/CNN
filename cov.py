# encoding: utf-8
# convolution layer

import math
import numpy as np
from im2col import col2im, im2col


class Cov:

    def __init__(self, kw, pad, s, out_c, inshape):
        self._kw = kw
        self._pad = pad
        self._s = s
        self._out_c = out_c
        self._inshape = inshape
        self._init_params = {'kw':kw, 'pad':pad, 's':s, 
                             'out_c':out_c, 'inshape':inshape}
        self.init_weight()
        self.get_outshape()
        self._use_col = True
    
    def init_weight(self, weight=None):
        if weight == None:
            in_c = self._inshape[0]
            size = (self._out_c, in_c, self._kw, self._kw)
            self._weight = np.random.normal(loc=0.0, scale=0.1, size=size)
        else:
            self._weight =np.array(weight)

    def get_params(self):
        params = {}
        params['class'] = 'Cov'
        params['weight'] = self._weight.tolist()
        params['init_params'] = self._init_params
        return params

    def get_outshape(self):
        in_c, in_h, in_w = self._inshape
        out_w = (in_w + 2 * self._pad - self._kw) / self._s + 1
        out_w = math.floor(out_w)
        out_h = (in_h + 2 * self._pad - self._kw) / self._s + 1
        out_h = math.floor(out_h)
        self._outshape = [self._out_c, out_h, out_w]
        return self._outshape

    def _padding(self, array):
        p_w = self._pad
        pad = ((0, 0), (0, 0), (p_w, p_w), (p_w, p_w))
        array = np.pad(array,  pad_width=pad, mode='constant', constant_values=0)
        return array

    def _reverse_padding(self, array):
        p_w = self._pad
        n, c, h, w = array.shape
        st_w = p_w
        st_h = p_w
        ed_w = w - p_w
        ed_h = h - p_w
        array = array[:, :, st_h:ed_h, st_w: ed_w]
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
        in_c = self._inshape[0]
        out_c, out_h, out_w = self._outshape
        pad_array = self._padding(in_array)
        self._pad_array_col = im2col(pad_array, self._kw, self._kw, 
                              self._s, self._inshape, self._outshape)
        weight_col = self._weight.reshape(out_c, -1)
        out_array = np.dot(weight_col, self._pad_array_col)
        out_array = out_array.reshape(out_c, out_h, out_w, n)
        out_array = out_array.transpose(3, 0, 1, 2)
        return out_array

    def _backward_col(self, in_grad, lr):
        n = in_grad.shape[0]
        out_c = self._outshape[0]
        ingrad_reshape = in_grad.transpose(1, 2, 3, 0).reshape(out_c, -1)
        w_delta = np.dot(ingrad_reshape, self._pad_array_col.T)
        w_delta = w_delta.reshape(self._weight.shape)

        weight_reshape = self._weight.reshape(out_c, -1)
        out_grad = np.dot(weight_reshape.T, ingrad_reshape)
        out_grad = col2im(out_grad, n, self._inshape, self._outshape, 
                          self._kw, self._kw, self._pad, self._s)

        self._weight -= w_delta * lr
        out_grad = self._reverse_padding(out_grad)
        return out_grad
    
    def _forward(self, in_array):
        n = in_array.shape[0]
        in_c = self._inshape[0]
        out_c, out_h, out_w = self._outshape
        self._pad_array = self._padding(in_array)
        out_array = np.zeros((n, out_c, out_h, out_w))
        for c_out in range(out_c):
            for c_in in range(in_c):
                for h in range(out_h):
                    for w in range(out_w):
                        st_w = self._s * w
                        st_h = self._s * h
                        ed_w = st_w + self._kw
                        ed_h = st_h + self._kw
                        sub_array = self._pad_array[:, c_in, st_h:ed_h, st_w:ed_w]
                        sub_weight = self._weight[c_out, c_in, :, :]
                        mul = sub_array *  sub_weight
                        mul_sum = np.sum(mul, axis=(1, 2))
                        out_array[:, c_out, h, w]  +=  mul_sum
        return out_array
    
    def _update(self, in_grad, lr):
        n = in_grad.shape[0]
        in_c = self._inshape[0]
        out_c, out_h, out_w = self._outshape 
        w_delta = np.zeros((self._out_c, in_c, self._kw, self._kw))
        for c_out in range(out_c):
            for c_in in range(in_c):
                for h  in range(out_h):
                    for w in range(out_w):
                        st_w = self._s * w
                        st_h = self._s * h
                        ed_w = st_w + self._kw
                        ed_h = st_h + self._kw
                        sub_array = self._pad_array[:, c_in, st_h:ed_h, st_w:ed_w]
                        sub_ingrad = in_grad[:, c_out, h, w].reshape(n, 1, 1)
                        sub_ingrad = np.broadcast_to(sub_ingrad, (n, self._kw, self._kw))
                        mul = sub_ingrad * sub_array
                        mul_sum = np.sum(mul, (0))
                        w_delta[c_out, c_in, :, :] += mul_sum
        self._weight = self._weight - lr * w_delta

    def _backward(self, in_grad, lr):
        n = in_grad.shape[0]
        in_c = self._inshape[0]
        out_grad = np.zeros_like(self._pad_array)
        out_c, out_h, out_w = self._outshape
        for c_out in range(out_c):
            for c_in in range(in_c):
                for h in range(out_h):
                    for w in range(out_w):
                        st_w = self._s * w
                        st_h = self._s * h
                        ed_w = st_w + self._kw
                        ed_h = st_h + self._kw
                        sub_weight = self._weight[c_out, c_in, :, :]
                        sub_ingrad = in_grad[ :, c_out, h, w ].reshape(n, 1, 1)
                        sub_ingrad = np.broadcast_to(sub_ingrad, (n, self._kw, self._kw))
                        mul = sub_weight * sub_ingrad
                        out_grad[:, c_in, st_h:ed_h, st_w:ed_w] += mul
        out_grad = self._reverse_padding(out_grad)
        self._update(in_grad, lr)
        return out_grad


if __name__ == '__main__':
    cov = Cov(3, 0, 1, 16, [3, 112, 112])
    outshape = cov.get_outshape()
    in_array = np.random.rand(128, 3, 112, 112)
    grad_shape = [128, 16, outshape[2], outshape[2]]
    in_grad = np.random.rand(128, 16, 110, 110)
    import time
    t1 = time.time()
    for i in range(10):
        print(cov._outshape)
        cov.forward(in_array)
    t2 = time.time()
    print((t2-t1)/10)
    t1 = time.time()
    for i in range(10):
        print(cov._outshape)
        cov.backward(in_grad, 0.1)
    t2 = time.time()
    print((t2-t1)/10)



