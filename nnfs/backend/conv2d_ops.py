# Backend operations for the Conv2d layer class.
# Uses Numba to make the operations run faster, it doesn't change the behavior of the functions.
# Assumes all inputs are pre-padded to work with Numba.

from numba import njit, prange
import numpy as np

# convolve kernel over image
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

# convolve kernel over delta -> derivative of delta wrt activation
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

# convolve delta over activation -> derivative of delta wrt kernel
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

# depthwise functions
@njit(parallel=True)
def dwise_conv2d(x, k, stride=1):
    N, C, H, W = x.shape
    C, ky, kx = k.shape

    ret_height = (H-ky) // stride + 1
    ret_width = (W-kx) // stride + 1
    ret = np.zeros((N, C, ret_height, ret_width))

    for n in prange(N):
        for c in range(C):
            for h in range(0, H-ky+1, stride):
                for w in range(0, W-kx+1, stride):
                    ret[n, c, h//stride, w//stride] = np.sum(k[c] * x[n, c, h:h+ky, w:w+kx])

    return ret

@njit(parallel=True)
def dwise_conv2d_backprop_wrt_a(delta, k):
    N, C, H, W = delta.shape
    C, ky, kx = k.shape

    ret = np.zeros((N, C, H-ky+1, W-kx+1))

    for n in prange(N):
        for c in range(C):
            for h in range(H-ky+1):
                for w in range(W-kx+1):
                    ret[n, c, h, w] = np.sum(k[c] * delta[n, c, h:h+ky, w:w+kx])

    return ret

@njit(parallel=True)
def dwise_conv2d_backprop_wrt_w(delta, x):
    N, C, H, W = delta.shape
    N, C, H_2, W_2 = x.shape
    ky = H_2 - H + 1
    kx = W_2 - W + 1

    ret = np.zeros((C, ky, kx))

    for c in prange(C):
        for h in range(ky):
            for w in range(kx):
                ret[c, h, w] = np.sum(delta[:, c] * x[:, c, h:h+H, w:w+W])

    return ret
