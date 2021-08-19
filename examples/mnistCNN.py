import sys, os
sys.path.append(os.path.abspath(os.path.join('..'))) # in order to access the nnfs module from this folder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from nnfs.layers import Conv2d, Flatten, FC
from nnfs.activations import SReLU, Linear
from nnfs.optimizers import SGD, Adam
from nnfs.models import Sequential
from nnfs.losses import CE
import matplotlib.pyplot as plt
import numpy.random as npr
import numpy as np

BATCH_SIZE = 32
EPOCHS = 3

(train_x, train_y), (test_x, test_y) = mnist.load_data()
m = train_x.shape[0]
train_x = train_x[:, np.newaxis, :, :] / 255.0 # NHW -> NCHW, then normalize`
train_y = to_categorical(train_y, num_classes=10)

test_x = test_x[:, np.newaxis, :, :] / 255.0 # NHW -> NCHW, then normalize`
test_y = to_categorical(test_y, num_classes=10)

m = BATCH_SIZE * (m // BATCH_SIZE)
train_x = train_x[:m]
train_y = train_y[:m]

in_n = train_x.shape[1] # 784
out_n = 10

for opt_class, col in [(SGD, 'blue'), (Adam, 'red')]:
    print('Training with:', opt_class)
    nn = Sequential()
    nn.add(Conv2d(16, in_n, stride=2, act_fn=SReLU))
    nn.add(Conv2d(8, 16, stride=2, act_fn=SReLU))
    nn.add(Conv2d(4, 8, stride=1, act_fn=SReLU))
    nn.add(Flatten())
    nn.add(FC(32, 49*4, act_fn=SReLU))
    nn.add(FC(out_n, 32, act_fn=Linear))

    opt = opt_class(nn.layers)
    nn.add_train_params(opt, CE())

    hist = nn.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    nn.evaluate(test_x, test_y, batch_size=BATCH_SIZE)
    plt.plot(hist['Epoch'], hist['Loss'], color=col)
    print()

plt.show()
