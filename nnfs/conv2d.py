import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from numba import njit, prange
import numpy as np
import time

m = 1
N = 4
kx = 3
ky = 3


@njit(parallel=True)
def conv2d(x, k, stride=1):
    N, _C, H, W = x.shape
    O, _I, ky, kx = k.shape

    ret_height = (H-ky) // stride + 1
    ret_width = (W-kx) // stride + 1
    ret = np.zeros((N, O, ret_height, ret_width))

    for n in prange(N):
        for o in range(O):
            for h in range(0, H-ky+1, stride):
                for w in range(0, W-kx+1, stride):
                    ret[n, o, h//stride, w//stride] = np.sum(k[o] * x[n, :, h:h+ky, w:w+kx])

    return ret

@njit(parallel=True)
def conv2d_backprop_wrt_a(delta, k):
    N, _C, H, W = delta.shape
    _O, I, ky, kx = k.shape

    ret = np.zeros((N, I, H-ky+1, W-kx+1))

    for n in prange(N):
        for i in range(I):
            for h in range(H-ky+1):
                for w in range(W-kx+1):
                    ret[n, i, h, w] = np.sum(k[:, i] * delta[n, :, h:h+ky, w:w+kx])

    return ret

@njit(parallel=True)
def conv2d_backprop_wrt_w(delta, x):
    N, O, H, W = delta.shape
    N, I, H_2, W_2 = x.shape
    ky = H_2 - H + 1
    kx = W_2 - W + 1

    ret = np.zeros((O, I, ky, kx))

    for o in prange(O):
        for i in range(I):
            for h in range(ky):
                for w in range(kx):
                    ret[o, i, h, w] = np.sum(delta[:, o] * x[:, i, h:h+H, w:w+W])

    return ret

class Conv2d:
    def __init__(self, out_f, in_f, ksize=3, stride=1, padding='same'):
        self.ksize = ksize
        self.stride = stride
        he_std = 1 / np.sqrt(ksize ** 2 * in_f)
        self.k = np.random.randn(out_f, in_f, ksize, ksize) * he_std
        #self.k = np.ones((out_f, in_f, ksize, ksize))

        if padding == 'same':
            _O, _I, ky, kx = self.k.shape
            y_pad_top = ky // 2 # Ex: ky = 3 -> pad_top = 1; ky = 4 -> pad_top = 2
            y_pad_bottom = (ky - 1) // 2 # Ex: ky = 3 -> pad_top = 1; ky = 4 -> pad_top = 1
            x_pad_left = kx // 2
            x_pad_right = (kx - 1) // 2
            self.padding = ((0, 0), (0, 0), (y_pad_top, y_pad_bottom), (x_pad_left, x_pad_right))
        else:
            self.padding = 'valid'

    def __call__(self, x):
        self.x = x

        if self.padding != 'valid':
            x_padded = np.pad(np.copy(x), self.padding)
            return conv2d(x_padded, self.k, self.stride)
        return conv2d(x, self.k, self.stride)
    
    def backward(self, delta, mode='w'):
        assert mode in ['w', 'a']

        _N, _C, H, W = delta.shape
        delta_strided = np.zeros((_N, _C, self.stride * H, self.stride * W))
        delta_strided[:, :, 0::self.stride, 0::self.stride] = delta

        _O, _I, ky, kx = self.k.shape

        if self.padding != 'valid':
            if mode == 'w':
                x_padded = np.pad(np.copy(self.x), self.padding)
                return conv2d_backprop_wrt_w(delta_strided, x_padded)
            elif mode == 'a':
                delta_padded = np.pad(np.copy(delta_strided), self.padding)
                return conv2d_backprop_wrt_a(delta_padded, self.k)

        else:
            if mode == 'w':
                return conv2d_backprop_wrt_w(delta_strided, self.x)
            else:
                y_pad, x_pad = (ky-1), (kx-1)
                delta_padded = np.pad(np.copy(delta_strided), ((0, 0), (0, 0), (y_pad, y_pad), (x_pad, x_pad)))
                return conv2d_backprop_wrt_a(delta_padded, self.k)

class GlobalPooling2d:
    def __init__(self, keepdims=True):
        self.keepdims = keepdims
        self.x = x
        return np.mean(x, axis=(2, 3), keepdims=self.keepdims)

    def backward(self):
        gain = 1 / (self.x.shape[2] * self.x.shape[3]) # 1 / num pixels
        return gain * np.ones_like(self.x)

inp = np.random.randn(m, 3, N, N)
lay_a = Conv2d(4, 3, stride=1)
lay_b = Conv2d(2, 4, stride=2)
lay_c = Conv2d(1, 2, stride=2)
end = GlobalPooling2d()
print(end(lay_c(lay_b(lay_a(inp)))).shape)

target = np.random.randn(*end(lay_c(lay_b(lay_a(inp)))).shape)
#target = np.mean(inp, axis=(1, 2, 3), keepdims=True)
print(target)
for i in range(20):
    a = lay_a(inp)
    b = lay_b(a)
    c = lay_c(b)
    d = end(c)
    #print('Loss:', np.sum(c))
    print('Loss:', (1 / 2) * np.mean(np.sum((d - target) ** 2, axis=(1, 2, 3))))
    #dl = np.ones_like(c)
    dl = (c - target) * end.backward()
    lr = 1e-3

    dc = lay_c.backward(dl, mode='a')
    grad_c = lay_c.backward(dl, mode='w')

    db = lay_b.backward(dc, mode='a')
    grad_b = lay_b.backward(dc, mode='w')

    da = lay_a.backward(db, mode='a')
    grad_a = lay_a.backward(db, mode='w')

    lay_c.k -= lr * grad_c / m
    lay_b.k -= lr * grad_b / m
    lay_a.k -= lr * grad_a / m

'''
#for nf, out_f in [(256, 128), (128, 64), (64, 32), (32, 16), (16, 8)]:
for nf, out_f in [(1, 1)]:
    a = np.ones((m, nf, N, N))
    #k = np.ones((out_f, nf, ky, kx))
    layer = Conv2d(out_f, nf, stride=2)

    start = time.time()
    #b = conv2d(a, k)
    b = layer(a)
    duration_conv = time.time() - start

    start = time.time()
    #b2 = conv2d_backprop(a, np.ones_like(b), mode='w')
    b2 = layer.backward(np.ones_like(b), mode='w')
    duration_conv_back_wrt_w = time.time() - start

    start = time.time()
    #b3 = conv2d_backprop(np.ones_like(b), k, mode='a')
    b3 = layer.backward(np.ones_like(b), mode='a')
    duration_conv_back_wrt_a = time.time() - start

    print()
    print('{} input filters | {} output filters'.format(nf, out_f))
    print('A shape:', a.shape)
    print('K shape:', layer.k.shape)
    print()
    print('Execution Time')
    print('  Conv2D:', duration_conv)
    print('  Conv2D Backprop (wrt K):', duration_conv_back_wrt_w)
    print('  Conv2D Backprop (wrt A):', duration_conv_back_wrt_a)
    print()
    print('Sum:', np.sum(a))
    print('A * K shape:', b.shape)
    print('Grad A * K wrt K shape:', b2.shape)
    print('Grad A * K wrt K:\n', b2)
    print('Grad A * K wrt A shape:', b3.shape)
    print('Grad A * K wrt A:\n', b3)
    assert b2.shape == layer.k.shape
    assert b3.shape == a.shape
    print('------------------------------------------')
    # hw - 49s
    # no hw - 55s
    # tf - 0.98s
    # torch - 0.8s
    # numba parallel no chw - 26s
    '''
