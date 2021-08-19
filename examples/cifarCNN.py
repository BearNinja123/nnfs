import sys, os
sys.path.append(os.path.abspath(os.path.join('..'))) # in order to access the nnfs module from this folder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from nnfs import layers, activations, optimizers, models, losses
from nnfs.activations import SReLU, Linear
import matplotlib.pyplot as plt
import numpy.random as npr
import numpy as np

BATCH_SIZE = 32
EPOCHS = 3

(train_x, train_y), (test_x, test_y) = cifar10.load_data()
m = train_x.shape[0]
train_x = np.transpose(train_x, (0, 3, 1, 2)) / 255.0 # NHWC -> NCHW, then normalize`
train_y = to_categorical(train_y, num_classes=10)

test_x = np.transpose(test_x, (0, 3, 1, 2)) / 255.0 # NHW -> NCHW, then normalize`
test_y = to_categorical(test_y, num_classes=10)

m = BATCH_SIZE * (m // BATCH_SIZE)
train_x = train_x[:m]
train_y = train_y[:m]

in_n = train_x.shape[1] # 3
out_n = 10

for opt_class, col in [(optimizers.Adam, 'blue'), (optimizers.Adam, 'red')]:
    print('Training with:', opt_class)
    nn = models.Sequential()

    nn.add(layers.Conv2d(16, in_n, ksize=1, act_fn=SReLU))
    nn.add(layers.DepthwiseConv2d(16, act_fn=SReLU))
    nn.add(layers.DepthwiseConv2d(16, act_fn=SReLU))
    nn.add(layers.Conv2d(16, 16, ksize=1, stride=1, act_fn=SReLU))
    nn.add(layers.Pooling2d())

    nn.add(layers.DepthwiseConv2d(16, act_fn=SReLU))
    nn.add(layers.DepthwiseConv2d(16, act_fn=SReLU))
    nn.add(layers.Conv2d(32, 16, ksize=1, stride=1, act_fn=SReLU))
    nn.add(layers.Pooling2d())

    nn.add(layers.DepthwiseConv2d(32, act_fn=SReLU))
    nn.add(layers.DepthwiseConv2d(32, act_fn=SReLU))
    nn.add(layers.Conv2d(16, 32, ksize=1, stride=1, act_fn=SReLU))
    nn.add(layers.Pooling2d())

    nn.add(layers.DepthwiseConv2d(16, act_fn=SReLU))
    nn.add(layers.DepthwiseConv2d(16, act_fn=SReLU))
    nn.add(layers.Conv2d(16, 16, ksize=1, act_fn=SReLU))

    nn.add(layers.Flatten())
    nn.add(layers.FC(32, 4*4*16, act_fn=SReLU))
    nn.add(layers.FC(out_n, 32, act_fn=Linear))

    opt = opt_class(nn.layers)
    nn.add_train_params(opt, losses.CE())

    hist = nn.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    nn.evaluate(test_x, test_y, batch_size=BATCH_SIZE)
    plt.plot(hist['Epoch'], hist['Loss'], color=col)
    print()

plt.show()
