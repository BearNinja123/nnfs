import numpy as np

class Loss:
    def __call__(self, y_pred, y_true):
        pass

    def backward(self):
        pass

class MSE(Loss):
    def __call__(self, y_pred, y_true):
        self.err = (y_pred - y_true)
        return np.mean(np.sum(self.err ** 2, axis=0))

    def backward(self):
        # I would divide the value by m but the backward step 
        # in the FC layer already takes the mean over each sample
        # so that actually takes care of that issue
        return 2 * self.err

# crossentropy with logits (softmax done on call, not before)
class CE(Loss):
    def __init__(self):
        super().__init__()
        self.reduce_dims = None
        self.m_axis = 0

    def __call__(self, y_pred, y_true):
        if self.reduce_dims is None:
            self.reduce_dims = set([i for i in range(len(y_pred.shape))])
            self.reduce_dims.remove(self.m_axis)
            self.reduce_dims = tuple(self.reduce_dims)

        # softmax
        self.y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=self.reduce_dims, keepdims=True)

        # crossentropy
        self.y_true = y_true
        return np.mean(-np.sum(y_true*np.log(self.y_pred), axis=self.reduce_dims))

    def backward(self):
        return self.y_pred - self.y_true # see https://www.youtube.com/watch?v=f-nW8cSa_Ec for proof
