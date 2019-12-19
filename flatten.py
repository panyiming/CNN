# encoding: utf-8
# flatten layer


class Flatten:

    def __init__(self, inshape):
        self._inshape = inshape
        self.update_weight = False
        self._init_params = {'inshape':inshape}
    
    def get_params(self):
        params = {}
        params = {'class':'Flatten'}
        params['init_params'] = self._init_params
        return params

    def get_outshape(self):
        in_c, in_h, in_w = self._inshape
        return in_c * in_h * in_w

    def forward(self, in_array):
        n, c, h, w = in_array.shape
        out_array = in_array.reshape(n, c*h*w)
        return out_array

    def backward(self, in_grad):
        n = in_grad.shape[0]
        c, h, w = self._inshape
        out_grad = in_grad.reshape(n, c, h, w)
        return out_grad
