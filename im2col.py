# encoding: utf-8
# speed up by im2col


import numpy as np


def im2col_idx(inshape, outshape, kw, kh, stride):
    inc, inh, inw = inshape
    outc, outh, outw = outshape
    h0 = np.repeat(np.arange(kh), kw)
    h0 = np.tile(h0, inc)
    h1 = stride * np.repeat(np.arange(outh), outw)
    w0 = np.tile(np.arange(kw), kh*inc)
    w1 = stride * np.tile(np.arange(outw), outh)
    hidx = h0.reshape(-1, 1) + h1.reshape(1, -1)
    widx = w0.reshape(-1, 1) + w1.reshape(1, -1)
    cidx = np.repeat(np.arange(inc), kw*kh).reshape(-1, 1)
    return cidx, hidx, widx


def im2col(pad_array, kw, kh, stride, inshape, outshape):
    cidx, hidx, widx = im2col_idx(inshape, outshape, kw, kh, stride)
    inc = inshape[0]
    cols = pad_array[:, cidx, hidx, widx]
    cols = cols.transpose(1, 2, 0).reshape(kw*kh*inc, -1)
    return cols


def col2im(cols, n, inshape, outshape, kh, kw, pad, stride):
    inc, inh, inw = inshape
    h_pad = inh + 2 * pad
    w_pad = inw + 2 * pad
    pad_zeros = np.zeros((n, inc, h_pad, w_pad))
    cidx, hidx, widx = im2col_idx(inshape, outshape, kw, kh, stride)
    cols_reshape = cols.reshape(inc*kw*kh, -1, n)
    cols_reshape = cols_reshape.transpose(2, 0, 1)
    np.add.at(pad_zeros, (slice(None), cidx, hidx, widx), cols_reshape)
    if pad == 0:
        return pad_zeros
    return pad_zeros[:, :, pad:-pad, pad:-pad]


if __name__ == '__main__':
    n = 4
    inshape = [3, 8, 8]
    kw = 2
    kh = 2
    s = 2
    pd = 1 
    outshape = [6, 5, 5]
    in_array = np.random.rand(n, 3, 10, 10)
    cols = im2col(in_array, kw, kh,s,  inshape, outshape)
    weight = np.random.rand(6, 3, 2, 2)
    weight = weight.reshape(6, -1)
    out = weight @ cols
    print(out.shape)
    out = out.reshape(6, 5, 5, 4)
    out = out.transpose(3, 0, 1, 2)
    print(out.shape)
    in_grad = np.random.rand(n, 6, 5, 5)
    in_grad_re = in_grad.transpose(1, 2, 3, 0).reshape(6, -1)
    dx_col = weight.T @ in_grad_re
    dx = col2im(dx_col, n, inshape, outshape, kh, kw, pd, s)
    print(dx.shape)
