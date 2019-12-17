# encoding: utf-8
# flatten layer


class Flatten(self):

    def __init__(self, inshape, batch_size):
        self._inshape = inshape
        self._batch_size = batch_size

    def forward(self, in_array):
        c, h, w = self._inshape
        n = self._batch_size
        out_array = in_array.reshape(n, c*h*w)
        return out_array

    def backward(self, in_grad):
        c, h, w = self._inshape
        n = self._batch_size
        out_grad = in_grad.reshape(n, c, h, w)
        return out_grad

