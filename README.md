# nnfs

A basic neural network framework similar to Pytorch and Keras made from Numpy ops. The framework includes fully connected layers and 2D convolution layers, L2 and crossentropy losses, and SGD (with Nesterov momentum) and Adam optimizers.

This framework is really slow (about 30x slower than normal frameworks on CPU) and doesn't run on the GPU at all, but was made for educational purposes.

Example datasets like MNIST, CIFAR10, and XOR are found in the [examples folder](/examples).
