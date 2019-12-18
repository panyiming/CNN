# encoding: utf-8
# flatten layer


class Flatten(self):

    def __init__(self, inshape):
        self._inshape = inshape
        self.update_weight = False
        self._init_params = {'inshape':inshape}
    
    def set_bs(self, batch_size=1):
        self._batch_size = batch_size
    
    def get_params(self):
        params = {'class':'Flatten'}
        params['init_params'] = self._init_params
        return params

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

