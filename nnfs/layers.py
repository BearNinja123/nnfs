from nnfs.backend.conv2d_ops import *
from nnfs.activations import Linear
from nnfs.misc import logg
from scipy.stats import truncnorm # truncated normal dist
import numpy.random as npr
import numpy as np

# layer class with trainable parameters and an activation
class WeightActLayer:
    def __init__(self, act_fn=Linear):
        self.w = None
        self.b = None
        self.act_fn = act_fn()
        self.inp = None

    def __call__(self, inputs):
        pass

    def backward(self, delta, wrt=None):
        pass

he_normal = lambda shape: (1 / np.sqrt(np.prod(shape[1:]))) * npr.randn(shape)
he_truncnorm = lambda shape: (1 / np.sqrt(np.prod(shape[1:]))) * truncnorm.rvs(-2, 2, size=shape) # normal dist with (min, max) = (-2, 2)

class FC(WeightActLayer):
    def __init__(self, out_f, in_f, act_fn=Linear):
        super().__init__(act_fn)
        self.out_f = out_f
        self.in_f = in_f

        he_std = (1 / np.sqrt(in_f))
        #self.w = npr.randn(out_f, in_f) * he_std
        self.w = he_truncnorm((out_f, in_f))

        self.b = np.zeros((1, out_f))

    def __call__(self, inputs):
        if self.w is None:
            in_f = inputs.shape[1]
            self.w = he_truncnorm((self.out_f, in_f))

        self.inp = inputs
        x = np.dot(inputs, self.w.T) + self.b
        x = self.act_fn(x)

        return x

    def backward(self, delta, wrt='w'):
        der_wrt_act = delta * self.act_fn.backward()

        if wrt == 'b':
            return np.mean(der_wrt_act, axis=0, keepdims=True)
        elif wrt == 'w':
            m = der_wrt_act.shape[0]
            return (1 / m) * np.dot(der_wrt_act.T, self.inp)
        elif wrt == 'a':
            return np.dot(der_wrt_act, self.w)

class Conv2d(WeightActLayer):
    def __init__(self, out_f, in_f, ksize=3, stride=1, padding='same', act_fn=Linear):
        super().__init__(act_fn)
        self.wsize = ksize
        self.stride = stride
        #he_std = 1 / np.sqrt(ksize ** 2 * in_f)
        #self.w = np.random.randn(out_f, in_f, ksize, ksize) * he_std
        self.ksize = ksize
        self.w = he_truncnorm((out_f, in_f, ksize, ksize))
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
        if self.w is None:
            in_f = inputs.shape[1]
            self.w = he_truncnorm((self.out_f, in_f, self.ksize, self.ksize))

        self.inp = inputs

        if self.padding != 'valid':
            x_padded = np.pad(np.copy(inputs), self.padding)
            z = conv2d(x_padded, self.w, self.stride) + self.b
        else:
            z = conv2d(inputs, self.w, self.stride) + self.b
        a = self.act_fn(z)

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

class DepthwiseConv2d(WeightActLayer):
    def __init__(self, in_f, ksize=3, stride=1, padding='same', act_fn=Linear):
        super().__init__(act_fn)
        self.wsize = ksize
        self.stride = stride
        #he_std = 1 / ksize
        #self.w = np.random.randn(in_f, ksize, ksize) * he_std
        self.w = he_truncnorm((in_f, ksize, ksize))
        self.b = np.zeros((1, in_f, 1, 1))

        if padding == 'same':
            _C, ky, kx = self.w.shape
            y_pad_top = ky // 2 # Ex: ky = 3 -> pad_top = 1; ky = 4 -> pad_top = 2
            y_pad_bottom = (ky - 1) // 2 # Ex: ky = 3 -> pad_top = 1; ky = 4 -> pad_top = 1
            x_pad_left = kx // 2
            x_pad_right = (kx - 1) // 2
            self.padding = ((0, 0), (0, 0), (y_pad_top, y_pad_bottom), (x_pad_left, x_pad_right))
        else:
            self.padding = 'valid'

    # all conv2d* functions are defined in nnfs/backend/conv2d_ops.py
    def __call__(self, inputs):
        if self.w is None:
            in_f = inputs.shape[1]
            self.w = he_truncnorm((self.out_f, in_f, self.ksize, self.ksize))

        self.inp = inputs

        if self.padding != 'valid':
            x_padded = np.pad(np.copy(inputs), self.padding)
            z = dwise_conv2d(x_padded, self.w, self.stride) + self.b
        else:
            z = dwise_conv2d(inputs, self.w, self.stride) + self.b
        a = self.act_fn(z)

        return a
    
    def backward(self, delta, wrt='w'):
        assert wrt in ['w', 'a', 'b']
        delta_wrt_act = delta * self.act_fn.backward()
        N, C, H, W = delta_wrt_act.shape

        if wrt == 'b':
            return np.mean(np.sum(delta, axis=(2, 3), keepdims=True), axis=0, keepdims=True)

        delta_strided = np.zeros((N, C, self.stride * H, self.stride * W))
        delta_strided[:, :, 0::self.stride, 0::self.stride] = delta_wrt_act

        _C, ky, kx = self.w.shape

        if self.padding != 'valid':
            if wrt == 'w':
                x_padded = np.pad(np.copy(self.inp), self.padding)
                return (1 / N) * dwise_conv2d_backprop_wrt_w(delta_strided, x_padded)
            elif wrt == 'a':
                delta_padded = np.pad(np.copy(delta_strided), self.padding)
                return dwise_conv2d_backprop_wrt_a(delta_padded, self.w)
        else:
            if wrt == 'w':
                return (1 / N) * dwise_conv2d_backprop_wrt_w(delta_strided, self.inp)
            else:
                y_pad, x_pad = (ky-1), (kx-1)
                delta_padded = np.pad(np.copy(delta_strided), ((0, 0), (0, 0), (y_pad, y_pad), (x_pad, x_pad)))
                return dwise_conv2d_backprop_wrt_a(delta_padded, self.w)

# either max or average pooling over spatial dimensions, assuming NCHW format
class Pooling2d:
    def __init__(self, mode='max', ksize=(2, 2)):
        assert type(ksize) in {tuple, int}
        assert mode in {'max', 'avg'}

        if isinstance(ksize, tuple):
            assert len(ksize) == 2
            self.kx, self.ky = ksize
        else:
            self.kx, self.ky = ksize, ksize

        self.mode = mode

    def __call__(self, inputs):
        self.inp = inputs
        N, C, H, W = inputs.shape
        ret_y = H // self.ky
        ret_x = W // self.kx

        new_shape = (N, C, ret_y, self.ky, ret_x, self.kx)
        reshaped = inputs.reshape(new_shape)

        if self.mode == 'max':
            ret = np.nanmax(reshaped, axis=(3, 5))
            self.output_inds = np.zeros_like(reshaped) # output_inds represents the effect of the input value at each index on the output
            self.output_inds[reshaped == ret[:, :, :, np.newaxis, :, np.newaxis]] = 1 # 0 if index is not a max, 1 otherwise
        else:
            ret = np.nanmean(reshaped, axis=(3, 5))
            self.output_inds = (1 / (self.kx * self.ky)) * np.ones_like(reshaped)

        return ret

    def backward(self, delta):
        reshaped_delta = delta[:, :, :, np.newaxis, :, np.newaxis]
        scaled_delta = self.output_inds * reshaped_delta
        ret = scaled_delta.reshape(self.inp.shape)

        return ret

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
