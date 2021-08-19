from nnfs.backend.conv2d_ops import *
from nnfs.activations import Linear
from nnfs.misc import logg
import numpy.random as npr
import numpy as np

# layer class with trainable parameters and an activation
class WeightActLayer:
    def __init__(self, act_fn=Linear):
        self.w = None
        self.b = None
        self.act_fn = act_fn()
        self.inp = None
        #self.z = None

    def __call__(self, inputs):
        pass

    def backward(self, delta, wrt=None):
        pass

class FC(WeightActLayer):
    def __init__(self, out_f, in_f, act_fn=Linear):
        super().__init__(act_fn)
        self.out_f = out_f
        self.in_f = in_f

        he_std = (1 / np.sqrt(in_f))
        self.w = npr.randn(in_f, out_f) * he_std

        self.b = np.zeros((1, out_f))

    def __call__(self, inputs):
        self.inp = inputs

        x = np.dot(inputs, self.w) + self.b

        x = self.act_fn(x)

        return x

    def backward(self, delta, wrt='w'):
        der_wrt_act = delta * self.act_fn.backward()

        if wrt == 'b':
            return np.mean(der_wrt_act, axis=0, keepdims=True)
        elif wrt == 'w':
            m = der_wrt_act.shape[1]
            return (1 / m) * np.dot(self.inp.T, der_wrt_act)
        elif wrt == 'a':
            return np.dot(der_wrt_act, self.w.T)

class Conv2d(WeightActLayer):
    def __init__(self, out_f, in_f, ksize=3, stride=1, padding='same', act_fn=Linear):
        super().__init__(act_fn)
        self.wsize = ksize
        self.stride = stride
        he_std = 1 / np.sqrt(ksize ** 2 * in_f)
        self.w = np.random.randn(out_f, in_f, ksize, ksize) * he_std
        self.b = np.zeros((1, out_f, 1, 1))

        if padding == 'same':
            _O, _I, ky, kx = self.w.shape
            y_pad_top = ky // 2 # Ex: ky = 3 -> pad_top = 1; ky = 4 -> pad_top = 2
            y_pad_bottom = (ky - 1) // 2 # Ex: ky = 3 -> pad_top = 1; ky = 4 -> pad_top = 1
            x_pad_left = kx // 2
            x_pad_right = (kx - 1) // 2
            self.padding = ((0, 0), (0, 0), (y_pad_top, y_pad_bottom), (x_pad_left, x_pad_right))
        else:
            self.padding = 'valid'

    # all conv2d* functions are defined in nnfs/backend/conv2d_ops.py
    def __call__(self, inputs):
        self.inp = inputs

        if self.padding != 'valid':
            x_padded = np.pad(np.copy(inputs), self.padding)
            #self.z = conv2d(x_padded, self.w, self.stride) + self.b
            z = conv2d(x_padded, self.w, self.stride) + self.b
        else:
            #self.z = conv2d(inputs, self.w, self.stride) + self.b
            z = conv2d(inputs, self.w, self.stride) + self.b
        #self.a = self.act_fn(self.z)
        #self.a = self.act_fn(z)
        a = self.act_fn(z)

        #return self.a
        return a
    
    def backward(self, delta, wrt='w'):
        assert wrt in ['w', 'a', 'b']
        delta_wrt_act = delta * self.act_fn.backward()
        N, C, H, W = delta_wrt_act.shape

        if wrt == 'b':
            return np.mean(np.sum(delta, axis=(2, 3), keepdims=True), axis=0, keepdims=True)

        delta_strided = np.zeros((N, C, self.stride * H, self.stride * W))
        delta_strided[:, :, 0::self.stride, 0::self.stride] = delta_wrt_act

        _O, _I, ky, kx = self.w.shape

        if self.padding != 'valid':
            if wrt == 'w':
                x_padded = np.pad(np.copy(self.inp), self.padding)
                return (1 / N) * conv2d_backprop_wrt_w(delta_strided, x_padded)
            elif wrt == 'a':
                delta_padded = np.pad(np.copy(delta_strided), self.padding)
                return conv2d_backprop_wrt_a(delta_padded, self.w)

        else:
            if wrt == 'w':
                return (1 / N) * conv2d_backprop_wrt_w(delta_strided, self.inp)
            else:
                y_pad, x_pad = (ky-1), (kx-1)
                delta_padded = np.pad(np.copy(delta_strided), ((0, 0), (0, 0), (y_pad, y_pad), (x_pad, x_pad)))
                return conv2d_backprop_wrt_a(delta_padded, self.w)

class GlobalPooling2d:
    def __init__(self, keepdims=False):
        self.keepdims = keepdims

    def __call__(self, inputs):
        self.inp = inputs
        return np.mean(inputs, axis=(2, 3), keepdims=self.keepdims)

    def backward(self, delta):
        # gain = delta / num pixels, delta.shape = (N, C)
        gain = (1 / (self.inp.shape[2] * self.inp.shape[3])) * delta
        return (gain[:, :, np.newaxis, np.newaxis] * np.ones_like(self.inp)) # self.x.shape = (N, C, H, W)

class Flatten:
    def __call__(self, inputs):
        self.input_shape = inputs.shape
        m = inputs.shape[0]
        return inputs.reshape(m, -1)

    def backward(self, delta):
        return delta.reshape(*self.input_shape)
