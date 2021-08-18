import numpy as np

# helper class so I replace "weight * a; bias * a" 
# with: "wbvar * a"
class WBVar:
    def __init__(self, w_arr, bias_arr):
        self.w = w_arr
        self.b = bias_arr

    def __add__(self, other):
        if isinstance(other, WBVar):
            return WBVar(self.w + other.w, self.b + other.b)
        else:
            return WBVar(self.w + other, self.b + other)

    def __neg__(self):
        return WBVar(-self.w, -self.b)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __mul__(self, other):
        if isinstance(other, WBVar):
            return WBVar(self.w * other.w, self.b * other.b)
        else:
            return WBVar(self.w * other, self.b * other)

    def __div__(self, other):
        if isinstance(other, WBVar):
            return WBVar(self.w / other.w, self.b / other.b)
        else:
            return WBVar(self.w / other, self.b / other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __pow__(self, other):
        if isinstance(other, WBVar):
            return WBVar(self.w ** other.w, self.b ** other.b)
        else:
            return WBVar(self.w ** other, self.b ** other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self.__add__(other.__neg__())

    def __rmul__(self, other):
        return self * other

    def __rdiv__(self, other):
        return self / other

    def __rtruediv__(self, other):
        return self / other

    def __rpow__(self, other):
        return self ** other

    def __iadd__(self, other):
        if isinstance(other, WBVar):
            self.w += other.w
            self.b += other.b
        else:
            self.w += other
            self.b += other

    def __isub__(self, other):
        if isinstance(other, WBVar):
            self.w -= other.w
            self.b -= other.b
        else:
            self.w -= other
            self.b -= other

    def __imul__(self, other):
        if isinstance(other, WBVar):
            self.w *= other.w
            self.b *= other.b
        else:
            self.w *= other
            self.b *= other

    def __idiv__(self, other):
        if isinstance(other, WBVar):
            self.w /= other.w
            self.b /= other.b
        else:
            self.w /= other
            self.b /= other

    def __itruediv__(self, other):
        self.__idiv__(other)

    def __ipow__(self, other):
        if isinstance(other, WBVar):
            self.w **= other.w
            self.b **= other.b
        else:
            self.w **= other
            self.b **= other

class Optimizer:
    def __init__(self, layers, lr=1e-3, auto_zero_grad=True):
        self.lr = lr
        self.layers = list(reversed(layers))
        self.auto_zero_grad = auto_zero_grad

        self.grads = self._setup_layer_vars()

    # setup a list with an empty WBVars of each layer, used for storing non-gradient data
    def _setup_layer_vars(self, layers=None):
        if layers is None:
            layers = self.layers

        wb_list = []
        for layer in layers:
            w_arr = np.zeros_like(layer.w)
            b_arr = np.zeros_like(layer.b)
            wb_list.append(WBVar(w_arr, b_arr))

        return wb_list

    def add_weight(self, weight, weight_list):
        weight_list.append(np.zeros_like(weight))

    def add_bias(self, bias, weight_list):
        weight_list.append(np.zeros_like(bias))

    def zero_grad(self):
        for param in self.grads:
            param *= 0

    def step(self):
        pass

class SGD(Optimizer):
    def __init__(
            self,
            layers,
            lr=1e-3, momentum=0.9, nag=True,
            weight_decay=0.0, decoupled=True,
            auto_zero_grad=True):

        super().__init__(layers, lr, auto_zero_grad)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.decoupled = decoupled
        self.nag = nag

        self.velocities = self._setup_layer_vars()

    def step(self):
        for idx, layer in enumerate(self.layers):
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
            layers,
            lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
            weight_decay=0.0, decoupled=True,
            auto_zero_grad=True):

        super().__init__(layers, lr, auto_zero_grad)
        self.beta1, self.beta2 = beta1, beta2
        self.weight_decay = weight_decay
        self.decoupled = decoupled
        self.eps = eps
        self.timestep = 0

        self.moment1_biased = self._setup_layer_vars()
        self.moment2_biased = self._setup_layer_vars()
        self.moment1_unbiased = self._setup_layer_vars()
        self.moment2_unbiased = self._setup_layer_vars()

    def step(self):
        self.timestep += 1

        for idx, layer in enumerate(self.layers):
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
            for param in self.grads:
                self.zero_grad()
