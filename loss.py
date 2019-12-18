# encoding: utf-8
# softmax layer


class Softmax:

    def __init__(self, class_num, batch_size):
        self._class_num = class_num
        self._batch_size = batch_size
        self._label_eye = np.eye(batch_size)

    def forward(self, in_array):
        x_exp = np.exp(in_array)
        x_sum = np.sum(x_exp, axis=1).reshape(batch_size, 1)
        x_sum = np.broadcast_to(x_sum, (8, 12))
        out_pro = x_exp / x_sum
        self._out_pro = out_pro
        return out_pro

    def backward(self, target):
        one_hot_target = self._label_eye[target]
        out_grad = self._out_pro - one_hot_target
        return out_grad

