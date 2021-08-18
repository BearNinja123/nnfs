from nnfs.activations import *
from nnfs.layers import *
from nnfs.losses import *
from tqdm import tqdm
import numpy.random as npr
import numpy as np

class Sequential:
    def __init__(self, layers=None, loss_fn=MSE()):
        if layers is None:
            self.layers = []
        else:
            assert isinstance(layers, list)
            self.layers = layers

        self.loss_fn = loss_fn

    def add_opt(self, opt):
        self.opt = opt

    def add(self, layer):
        self.layers.append(layer)

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)

        return x

    def calc_loss(self, y_pred, y_true):
        loss = self.loss_fn(y_pred, y_true)
        return loss

    def backward(self):
        delta = self.loss_fn.backward()

        for idx, layer in enumerate(list(reversed(self.layers))):
            self.opt.grads[idx].w += layer.backward(delta, wrt='w')
            self.opt.grads[idx].b += layer.backward(delta, wrt='b')
            delta = layer.backward(delta, wrt='a')

    def fit(self, X, y, epochs=1, batch_size=None, m_axis=1):
        m = X.shape[m_axis]
        if batch_size is None:
            batch_size = m
        num_batches = m // batch_size

        X_batched = np.split(X, num_batches, axis=m_axis)
        y_batched = np.split(y, num_batches, axis=m_axis)

        cost_history = {
                'Epoch': np.arange(epochs),
                'Cost': np.empty((epochs,)),
                'Accuracy': np.empty((epochs,)),
                }

        for epoch in range(epochs):
            epoch_cost = 0
            epoch_acc = 0
            batch_idx = 0

            progress_bar = tqdm(zip(X_batched, y_batched), total=num_batches)
            for batch_x, batch_y in progress_bar:
                y_pred = self(batch_x)

                # calc loss and accuracy
                loss = self.loss_fn(y_pred, batch_y)
                epoch_cost += loss
                y_correct = (np.argmax(y_pred, axis=1-m_axis) == np.argmax(batch_y, axis=1-m_axis)).astype(np.int32)
                accuracy = np.mean(y_correct)
                epoch_acc += accuracy

                self.backward()
                self.opt.step()

                batch_idx += 1
                progress_bar.set_description('Epoch: {} | Loss: {} | Accuracy: {}'.format(epoch, round(epoch_cost / batch_idx, 4), round(epoch_acc / batch_idx, 4)))

            cost_history['Cost'][epoch] = epoch_cost / num_batches
            cost_history['Accuracy'][epoch] = epoch_acc / num_batches

        return cost_history

    def evaluate(self, X, y, batch_size=None, m_axis=1):
        m = batch_size * (X.shape[m_axis] // batch_size)

        if m_axis == 0:
            X = X[:m]
            y = y[:m]
        else:
            X = X[:, :m]
            y = y[:, :m]

        if batch_size is None:
            batch_size = m
        num_batches = m // batch_size

        X_batched = np.split(X, num_batches, axis=m_axis)
        y_batched = np.split(y, num_batches, axis=m_axis)

        metrics = {
                'Cost': None,
                'Accuracy': None,
                }

        cost = 0
        acc = 0
        batch_idx = 0

        progress_bar = tqdm(zip(X_batched, y_batched), total=num_batches)
        for batch_x, batch_y in progress_bar:
            y_pred = self(batch_x)

            # calc loss and accuracy
            loss = self.loss_fn(y_pred, batch_y)
            cost += loss
            y_correct = (np.argmax(y_pred, axis=1-m_axis) == np.argmax(batch_y, axis=1-m_axis)).astype(np.int32)
            accuracy = np.mean(y_correct)
            acc += accuracy

            batch_idx += 1
            progress_bar.set_description('Evaluation | Loss: {} | Accuracy: {}'.format(round(cost / batch_idx, 4), round(acc / batch_idx, 4)))

        metrics['Cost'] = cost / num_batches
        metrics['Accuracy'] = acc / num_batches

        return metrics

class MLP(Sequential):
    def __init__(self, input_neurons, layer_neurons, output_act=Linear(), intermediate_act=SReLU(), loss_fn=MSE()):
        super().__init__([] loss_fn) # init with empty list, add layers in _setup_layers method

        self.input_neurons = input_neurons
        self.layer_neurons = layer_neurons
        self._setup_layers()
        self.layers[-1].act_fn = output_act

        self.intermediate_act = intermediate_act
        self.output_act = output_act

    def _setup_layers(self):
        for idx, neuron_count in enumerate(self.layer_neurons):
            if idx == 0:
                self.add(FC(neuron_count, self.input_neurons, act_fn=self.intermediate_act))
            else:
                self.add(FC(neuron_count, self.layer_neurons[idx-1], act_fn=self.intermediate_act))

class CNN(Sequential):
    def __init__(self, input_filters, layer_filters, output_act=Linear(), intermediate_act=SReLU(), loss_fn=MSE()):
        super().__init__([] loss_fn) # init with empty list, add layers in _setup_layers method

        self.input_filters = input_filters
        self.layer_filters = layer_filters
        self._setup_layers()
        self.layers[-1].act_fn = output_act

        self.intermediate_act = intermediate_act
        self.output_act = output_act

    def _setup_layers(self):
        for idx, filter_count in enumerate(self.layer_filters):
            if idx == 0:
                self.add(Conv2d(filter_count, self.input_shape, act_fn=self.intermediate_act))
            else:
                self.add(Conv2d(filter_count, self.layer_filters[idx-1], act_fn=self.intermediate_act))
