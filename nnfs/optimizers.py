from nnfs.layers import WeightActLayer
from nnfs.backend.wbvar import WBVar
from typing import List
import numpy as np

class Optimizer:
    def __init__(self, lr=1e-3, auto_zero_grad=True):
        self.lr = lr
        self.layers = None
        self.grads = None
        self.auto_zero_grad = auto_zero_grad

    def set_params(self, layers: list):
        # layers are reversed since the last layers receive the first gradients during backprop
        self.layers = list(reversed(layers))
        self.grads = self._setup_layer_vars()

    # setup a list with an empty WBVars of each layer, used for storing gradient-like data
    def _setup_layer_vars(self, layers: List = None) -> List[WBVar]:
        if layers is None:
            layers = self.layers

        wb_list = []
        for layer in layers:
            if not isinstance(layer, WeightActLayer): # if layer has no trainable params, continue
                wb_list.append(None)
                continue

            w_arr = np.zeros_like(layer.w)
            b_arr = np.zeros_like(layer.b)
            wb_list.append(WBVar(w_arr, b_arr))

        return wb_list

    def zero_grad(self):
        for param in self.grads:
            if not isinstance(param, WBVar):
                continue
            param *= 0

    def step(self):
        pass

class SGD(Optimizer):
    def __init__(
            self,
            lr=1e-3, momentum=0.9, nag=True,
            weight_decay=0.0, decoupled=True,
            auto_zero_grad=True):

        super().__init__(lr, auto_zero_grad)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.decoupled = decoupled
        self.nag = nag

    def set_params(self, layers: list):
        super().set_params(layers)
        self.velocities = self._setup_layer_vars()

    def step(self):
        for idx, layer in enumerate(self.layers):
            if not isinstance(layer, WeightActLayer):
                continue

            if self.weight_decay > 0:
                w_decay = self.weight_decay * layer.w
                if not self.decoupled:
                    self.grads[idx].w += w_decay
            else:
                w_decay = 0

            if self.momentum > 0:
                self.velocities[idx] = self.momentum * self.velocities[idx] + self.lr * self.grads[idx]

                if self.nag:
                    self.grads[idx] = self.momentum * self.velocities[idx] + self.lr * self.grads[idx]
                else:
                    self.grads[idx] = self.velocities[idx]

                layer.w -= self.grads[idx].w
                layer.b -= self.grads[idx].b

            else:
                layer.w -= self.lr * self.grads[idx].w
                layer.b -= self.lr * self.grads[idx].b
            
            if self.weight_decay > 0 and self.decoupled:
                layer.w -= w_decay

        if self.auto_zero_grad:
            self.zero_grad()

class Adam(Optimizer):
    def __init__(
            self,
            lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
            weight_decay=0.0, decoupled=True,
            auto_zero_grad=True):

        super().__init__(lr, auto_zero_grad)
        self.beta1, self.beta2 = beta1, beta2
        self.weight_decay = weight_decay
        self.decoupled = decoupled
        self.eps = eps
        self.timestep = 0

    def set_params(self, layers: list):
        super().set_params(layers)
        self.moment1_biased = self._setup_layer_vars()
        self.moment2_biased = self._setup_layer_vars()
        self.moment1_unbiased = self._setup_layer_vars()
        self.moment2_unbiased = self._setup_layer_vars()

    def step(self):
        self.timestep += 1

        for idx, layer in enumerate(self.layers):
            if not isinstance(layer, WeightActLayer):
                continue

            if self.weight_decay > 0:
                w_decay = self.weight_decay * layer.w
                if not self.decoupled:
                    self.grads[idx].w += w_decay
            else:
                w_decay = 0

            self.moment1_biased[idx] = self.beta1 * self.moment1_biased[idx] + (1 - self.beta1) * self.grads[idx]
            self.moment1_unbiased[idx] = (1 / (1 - self.beta1 ** self.timestep)) * self.moment1_biased[idx]

            self.moment2_biased[idx] = self.beta2 * self.moment2_biased[idx] + (1 - self.beta2) * (self.grads[idx] ** 2)
            self.moment2_unbiased[idx] = (1 / (1 - self.beta2 ** self.timestep)) * self.moment2_biased[idx]

            self.grads[idx] = self.lr * self.moment1_unbiased[idx] / (self.moment2_unbiased[idx] ** 0.5 + self.eps)

            layer.w -= self.grads[idx].w
            layer.b -= self.grads[idx].b

            if self.weight_decay > 0 and self.decoupled:
                layer.w -= w_decay

        if self.auto_zero_grad:
            self.zero_grad()

# SGD with adaptive gradient clipping
class SGD_AGC(SGD):
    def __init__(
            self, 
            lr=1e-1, momentum=0.9, nag=True,
            weight_decay=0.0, decoupled=True,
            lb=1e-2, eps=1e-3,
            auto_zero_grad=True):

        super().__init__(
                lr, momentum, nag,
                weight_decay, decoupled,
                auto_zero_grad)

        self.lb = lb # gradient clipping param
        self.eps = eps # value to allow zero-init values to change

    def unitwise_norm(self, x):
        rank = len(x.shape)
        reduce_axes_map = {0: None, 1: None, 2: 0, 3: 0, 4: (0, 1, 2)}
        axis = reduce_axes_map[rank]
        keepdims = rank > 1
        return np.sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5

    def agc(self, param, grad):
        param_norm = np.maximum(self.unitwise_norm(param), self.eps)
        grad_norm = self.unitwise_norm(grad)
        max_norm = param_norm * self.lb
        # If grad norm > clipping * param_norm, rescale
        trigger = grad_norm > max_norm
        clipped_grad = grad * (max_norm / np.maximum(grad_norm, 1e-6))
        grad = np.where(trigger, clipped_grad, grad)

        return grad

    def step(self):
        # perform AGC
        for layer_idx, layer in enumerate(self.layers):
            # layer_idx == 0 to signify last FC layer is hacky but should work if the model is reasonable
            if not isinstance(layer, WeightActLayer) or layer_idx == 0:
                continue

            # clipping for weights
            w_grad_clipped = self.agc(layer.w, self.grads[layer_idx].w)
            b_grad_clipped = self.agc(layer.b, self.grads[layer_idx].b)
            self.grads[layer_idx] = WBVar(w_grad_clipped, b_grad_clipped)
        
        # run the step on the modified grads
        return super().step()
