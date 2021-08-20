import sys, os
sys.path.append(os.path.abspath(os.path.join('..'))) # in order to access the nnfs module from this folder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from nnfs import optimizers, activations, models, losses
from nnfs.activations import SReLU
import matplotlib.pyplot as plt
import numpy.random as npr
import numpy as np

BATCH_SIZE = 256
EPOCHS = 3

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

for opt_class, col in [(optimizers.SGD_AGC, 'green'), (optimizers.SGD, 'blue'), (optimizers.Adam, 'red')]:
    print('Training with:', opt_class)
    nn = models.MLP(in_n, [256, 128, 64, 32, 16, out_n], intermediate_act=SReLU)
    opt = opt_class()
    nn.build((in_n,), opt, losses.CE())

    hist = nn.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    nn.evaluate(test_x, test_y, batch_size=BATCH_SIZE)
    plt.plot(hist['Epoch'], hist['Loss'], color=col)
    print()

plt.show()
