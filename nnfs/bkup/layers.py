from nnfs.backend.conv2d_ops import *
from nnfs.activations import Linear
import numpy.random as npr
import numpy as np

class Layer:
    def __init__(self, act_fn=Linear()):
        self.w = None
        self.b = None
        self.act_fn = act_fn
        self.inp = None
        self.z = None
        #self.a = None

    def __call__(self, inputs):
        pass

    def backward(self, delta, wrt=None):
        pass

class FC(Layer):
    def __init__(self, out_f, act_fn=Linear()):
        super().__init__(act_fn)
        self.out_f = out_f
        self.in_f = in_f

        he_std = (1 / np.sqrt(in_f))
        self.w = npr.randn(out_f, in_f) * he_std

        self.b = np.zeros((out_f, 1))

    def __call__(self, inputs):
        self.inp = inputs

        x = np.dot(self.w, inputs) + self.b
        self.z = np.copy(x)

        x = self.act_fn(x)

        return x

    def backward(self, delta, wrt='w'):
        der_wrt_act = delta * self.act_fn.backward()

        if wrt == 'b':
            return np.mean(der_wrt_act, axis=1, keepdims=True)
        elif wrt == 'w':
            m = der_wrt_act.shape[1]
            return (1 / m) * np.dot(der_wrt_act, self.inp.T)
        elif wrt == 'a':
            return np.dot(self.w.T, der_wrt_act)

class Conv2d(Layer):
    def __init__(self, out_f, in_f, ksize=3, stride=1, padding='same', act_fn=Linear()):
        super().__init__(act_fn)
        self.wsize = ksize
        self.stride = stride
        he_std = 1 / np.sqrt(ksize ** 2 * in_f)
        self.w = np.random.randn(out_f, in_f, ksize, ksize) * he_std
        self.b = np.zeros((out_f,))

        if padding == 'same':
            _O, _I, ky, kx = self.w.shape
            y_pad_top = ky // 2 # Ex: ky = 3 -> pad_top = 1; ky = 4 -> pad_top = 2
            y_pad_bottom = (ky - 1) // 2 # Ex: ky = 3 -> pad_top = 1; ky = 4 -> pad_top = 1
            x_pad_left = kx // 2
            x_pad_right = (kx - 1) // 2
            self.padding = ((0, 0), (0, 0), (y_pad_top, y_pad_bottom), (x_pad_left, x_pad_right))
        else:
            self.padding = 'valid'

    def __call__(self, inputs):
        self.inp = x

        if self.padding != 'valid':
            x_padded = np.pad(np.copy(x), self.padding)
            self.z = conv2d(x_padded, self.w, self.stride) + self.b
        self.z = conv2d(x, self.w, self.stride) + self.b
        self.a = self.act_fn(self.z)

        return self.a
    
    def backward(self, delta, mode='w'):
        assert mode in ['w', 'a', 'b']
        delta_wrt_act = delta * self.act_fn.backward()
        N, C, H, W = delta_wrt_act.shape

        if mode == 'b':
            return np.mean(np.sum(delta, axis=(2, 3)), axis=0)

        delta_strided = np.zeros((N, C, self.stride * H, self.stride * W))
        delta_strided[:, :, 0::self.stride, 0::self.stride] = delta_wrt_act

        _O, _I, ky, kx = self.w.shape

        if self.padding != 'valid':
            if mode == 'w':
                x_padded = np.pad(np.copy(self.z), self.padding)
                return (1 / N) * conv2d_backprop_wrt_w(delta_strided, x_padded)
            elif mode == 'a':
                delta_padded = np.pad(np.copy(delta_strided), self.padding)
                return conv2d_backprop_wrt_a(delta_padded, self.w)

        else:
            if mode == 'w':
                return (1 / N) * conv2d_backprop_wrt_w(delta_strided, self.z)
            else:
                y_pad, x_pad = (ky-1), (kx-1)
                delta_padded = np.pad(np.copy(delta_strided), ((0, 0), (0, 0), (y_pad, y_pad), (x_pad, x_pad)))
                return conv2d_backprop_wrt_a(delta_padded, self.w)

class GlobalPooling2d:
    def __init__(self, keepdims=True):
        self.keepdims = keepdims
        self.x = x
        return np.mean(x, axis=(2, 3), keepdims=self.weepdims)

    def backward(self):
        gain = 1 / (self.x.shape[2] * self.x.shape[3]) # 1 / num pixels
        return gain * np.ones_like(self.x)
