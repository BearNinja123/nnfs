import sys, os
sys.path.append(os.path.abspath(os.path.join('..'))) # in order to access the nnfs module from this folder

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, normalize
from nnfs.activations import *
from nnfs.optimizers import *
from nnfs.layers import *
from nnfs.models import *
from nnfs.losses import *
import matplotlib.pyplot as plt
import numpy.random as npr
import numpy as np

BATCH_SIZE = 32
EPOCHS = 50

(train_x, train_y), (test_x, test_y) = mnist.load_data()
m = train_x.shape[0]
train_x = train_x[:, np.newaxis, :, :] / 255.0 # NHW -> NCHW, then normalize`
print(train_x.shape)
train_y = to_categorical(train_y, num_classes=10)

test_x = test_x[:, np.newaxis, :, :] / 255.0 # NHW -> NCHW, then normalize`
test_y = to_categorical(test_y, num_classes=10)

m = BATCH_SIZE * (m // BATCH_SIZE)
train_x = train_x[:m]
train_y = train_y[:m]

in_n = train_x.shape[1] # 784
out_n = 10

for opt_class, col in [(SGD, 'blue'), (Adam, 'red')]:
    #nn = CNN(in_n, [16, 8, 4, out_n], strides=[2, 2, 1, 1], intermediate_act=SReLU, loss_fn=CE())
    nn = Sequential(loss_fn=CE())
    nn.add(Conv2d(16, in_n, stride=2, act_fn=SReLU))
    nn.add(Conv2d(8, 16, stride=2, act_fn=SReLU))
    nn.add(Conv2d(4, 8, stride=1, act_fn=SReLU))
    nn.add(Flatten())
    nn.add(FC(32, 49*4, act_fn=SReLU))
    nn.add(FC(10, 32, act_fn=Linear))

    opt = opt_class(nn.layers)
    nn.add_opt(opt)

    hist = nn.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    nn.evaluate(test_x, test_y, batch_size=BATCH_SIZE)
    print(train_y[:, :1])
    print(softmax(nn(train_x[:, :1])))
    plt.plot(hist['Epoch'], hist['Cost'], color=col)

plt.show()
