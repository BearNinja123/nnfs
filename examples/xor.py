import sys, os
sys.path.append(os.path.abspath(os.path.join('..'))) # in order to access the nnfs module from this folder

from nnfs.activations import *
from nnfs.optimizers import *
from nnfs.layers import *
from nnfs.models import *
from nnfs.losses import *
import matplotlib.pyplot as plt
import numpy.random as npr
import numpy as np

m = 256
n = 2
out_n = 2

X = npr.randint(0, 2, (m, n))
y = np.zeros((m, out_n))
xor_idxs = np.logical_xor(X[:, 0], X[:, 1]).astype(np.int8)
for i in range(m):
    y[i, xor_idxs[i]] = 1

nn = MLP(n, [2, out_n], output_act=Linear, intermediate_act=Sigmoid, loss_fn=CE())
opt = SGD(nn.layers, lr=1e-2, momentum=0.99, nag=True)
nn.add_opt(opt)

hist = nn.fit(X, y, epochs=50, batch_size=4)
plt.plot(hist['Epoch'], hist['Cost'])
plt.show()
