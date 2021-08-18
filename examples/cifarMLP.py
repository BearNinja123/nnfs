import sys, os
sys.path.append(os.path.abspath(os.path.join('..'))) # in order to access the nnfs module from this folder

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical, normalize
from nnfs.activations import *
from nnfs.optimizers import *
from nnfs.layers import *
from nnfs.models import *
from nnfs.losses import *
import matplotlib.pyplot as plt
import numpy.random as npr
import numpy as np

BATCH_SIZE = 128
EPOCHS = 100

(train_x, train_y), (test_x, test_y) = cifar10.load_data()
m = train_x.shape[0]
train_x = train_x.reshape(m, -1) / 255.0
train_y = to_categorical(train_y, num_classes=10)

test_x = test_x.reshape(test_x.shape[0], -1) / 255.0
test_y = to_categorical(test_y, num_classes=10)

m = BATCH_SIZE * (m // BATCH_SIZE)
train_x = train_x[:m]
train_y = train_y[:m]

in_n = train_x.shape[1] # 3072
out_n = 10

for opt_class, col in [(SGD, 'blue'), (Adam, 'red')]:
    nn = MLP(in_n, [256, 128, 64, 32, 16, out_n], intermediate_act=SReLU, loss_fn=CE())
    print(opt_class)
    opt = opt_class(nn.layers, lr=1e-3)
    nn.add_opt(opt)

    hist = nn.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    nn.evaluate(test_x, test_y, batch_size=BATCH_SIZE)
    print(train_y[:, :1])
    print(softmax(nn(train_x[:, :1])))
    plt.plot(hist['Epoch'], hist['Cost'], color=col)

plt.show()
