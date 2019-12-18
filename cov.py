# encoding: utf-8
# convolution layer

import math
import numpy as np


class Cov:

    def __init__(self, kw, pad, s, out_c, inshape):
        self._kw = kw
        self._pad = pad
        self._s = s
        self._out_c = out_c
        self.init_weight()
        self._inshape = inshape
        self.update_weight = True
        self._init_params = {'kw':kw, 'pad':pad, 's':s, 
                             'out_c':out_c, 'inshape':inshape}

    def set_bs(self, batch_size=1):
        self._batch_size = batch_size
        out_shape = self._get_outshape()
        out_shape.insert(0, batch_size)
        self.out_array = np.zeros(out_shape)
    
    def init_weight(self, weight=None):
        if weight == None:
            self.weight = np.random.rand(self._out_c, self._kw, self._kw)
        else:
            self.weight =np.array(weight)

    def get_params(self):
        params['class'] = 'Cov'
        params['weight'] = self.weight.tolist()
        params['init_params'] = self._init_params
        return params

    def _get_outshape(self):
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
        _, out_c, out_h, out_w = self._outshape
        pad_array = self._padding(in_array)
        self.pad_array = pad_array
        for c in range(out_c):
            for h in range(out_h):
                for w in range(out_w):
                    st_w = self._s * w
                    st_h = self._s * h
                    ed_w = st_w + self._kw
                    ed_h = st_h + self._kw
                    mul = pad_array[:, :, st_h:ed_h, st_w:ed_w] * self.weight[c, :, :]
                    mul_sum = np.sum(mul, axis=(1, 2, 3))
                    self.out_array[:, c, h, w] =  mul_sum
        return self.out_array

    def update(self, in_grad, lr):
        n, out_c, out_h, out_w = self._outshape 
        w_delta = np.zeros((self._out_c, self._kw, self._kw))
        for c in range(out_c):
            for h  in range(out_h):
                for w in range(out_w):
                    st_w = self._s * w
                    st_h = self._s * h
                    ed_w = st_w + self._kw
                    ed_h = st_h + self._kw
                    mul = in_grad[:, c, h, w].reshape(n, 1, 1, 1) * self.pad_array[:, :, 
                          st_h:ed_h, st_w:ed_w]
                    mul_sum = np.sum(mul, (0, 1))
                    w_delta[c, :, :] += mul_sum
        self.weight = self.weight - lr * w_delta

    def backward(self, in_grad):
        out_grad = np.zeros_like(self.pad_array)
        n, out_c, out_h, out_w = self._outshape
        for c in range(out_c):
            for h in range(out_h):
                for w in range(out_w):
                    st_w = self._s * w
                    st_h = self._s * h
                    ed_w = st_w + self._kw
                    ed_h = st_h + self._kw
                    out_grad[:, :, st_h:ed_h, st_w:ed_w] += \
                            self.weight[c, :, :] * in_grad[ :, c, h, w].reshape(n, 1, 1, 1)
        out_grad = self._reverse_padding(out_grad)
        return out_grad
        

if __name__ == '__main__':
    inshape = [3, 112, 96]
    conv = Cov(3, 1, 2, 8,  inshape)
    conv.set_bs(16)
    in_array = np.random.rand(16, 3, 112, 96)
    in_grad = np.random.rand(16, 8, 56, 48)
    out_array = conv.forward(in_array)
    print(out_array.shape)
    out_grad = conv.backward(in_grad)
    print(out_grad.shape)
    conv.update(in_grad, lr=0.1)
    import pickle
    cov = pickle.dumps(conv)
    import json
    with open('test.json', 'w') as f:
        f.write(str(cov)+'\n')

