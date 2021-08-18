import numpy as np

class Activation:
    def __call__(self, x):
        pass

    def backward(self):
        pass

class ReLU(Activation):
    def __call__(self, x):
        self.act = np.maximum(0, x)
        return self.act

    def backward(self):
        return np.sign(self.act)

class SReLU(ReLU):
    def __init__(self, gain=np.sqrt(2)):
        super().__init__()
        self.gain = gain

    def __call__(self, x):
        self.act = self.gain * np.maximum(0, x)
        return self.act

    def backward(self):
        return self.gain * np.sign(self.act)

class Linear(Activation):
    def __call__(self, x):
        self.act = x
        return self.act

    def backward(self):
        return 1

class Sigmoid(Activation):
    def __call__(self, x):
        big = 100
        self.act = 1 / (1 + np.exp(np.minimum(-x, big)))
        return self.act

    def backward(self):
        return self.act * (1 - self.act)

# not an Activation subclass because of its nasty jacobian
def softmax(x, m_axis=1):
        reduce_dims = set([i for i in range(len(x.shape))])
        reduce_dims.remove(m_axis)
        reduce_dims = tuple(reduce_dims)
        act = np.exp(x) / np.sum(np.exp(x), axis=reduce_dims)
        return act
