from nnfs import layers, metrics, activations, losses
from nnfs.misc import logg
from typing import List, Tuple
from tqdm import tqdm
import numpy.random as npr
import numpy as np

class Sequential:
    def __init__(self, layers: List = None):
        if layers is None: # i would default layers to [] instead of None but that doesn't always set layers to an empty list for some reason?
            self.layers = []
        else:
            self.layers = layers 

        self.metrics = []
        self.loss_fn = None
        self.opt = None

    # create the weights and biases of the model, set up optimizers, loss functions, and metrics
    def build(self, input_shape: Tuple[int], optimizer, loss_fn=losses.MSE(), metric_list: List[str] = ['loss', 'accuracy']):
        x = np.zeros((1, *input_shape))
        for layer in self.layers:
            x = layer(x)

        metric_aliases = {
            'loss': metrics.LossMetric(),
            'accuracy': metrics.Accuracy(),
            'acc': metrics.Accuracy(),
        }

        optimizer.set_params(self.layers)
        self.opt = optimizer
        self.loss_fn = loss_fn

        for metric_name in metric_list:
            if metric_name in metric_aliases:
                self.metrics.append(metric_aliases[metric_name])
            else:
                self.metrics.append(metric_name)

    def add_loss(self, loss_fn):
        self.loss_fn = loss_fn

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
            if isinstance(layer, layers.WeightActLayer):
                self.opt.grads[idx].w += layer.backward(delta, wrt='w')
                self.opt.grads[idx].b += layer.backward(delta, wrt='b')
                delta = layer.backward(delta, wrt='a')
            else:
                delta = layer.backward(delta)

    def fit(self, X, y, epochs=1, batch_size=None, train=True):
        m = batch_size * (X.shape[0] // batch_size)

        X = X[:m]
        y = y[:m]

        if batch_size is None:
            batch_size = m
        num_batches = m // batch_size

        X_batched = np.split(X, num_batches, axis=0)
        y_batched = np.split(y, num_batches, axis=0)

        cost_history = {'Epoch': np.arange(epochs)}
        for metric in self.metrics:
            cost_history[metric.name] = np.empty((epochs,))

        for epoch in range(epochs):
            batch_idx = 0
            for metric in self.metrics:
                metric.reset()

            progress_bar = tqdm(zip(X_batched, y_batched), total=num_batches)
            for batch_x, batch_y in progress_bar:
                if train:
                    tqdm_string = 'Epoch: {}'.format(epoch)
                else:
                    tqdm_string = 'Evaluate'
                batch_idx += 1

                y_pred = self(batch_x)

                # calc metrics
                loss = self.loss_fn(y_pred, batch_y)

                for metric in self.metrics:
                    if metric.name == 'Loss':
                        metric.update(loss)
                    else:
                        metric.update(y_pred, batch_y)
                    tqdm_string += metric.disp(batch_idx)

                if train:
                    self.backward()
                    self.opt.step()

                progress_bar.set_description(tqdm_string)

            for metric in self.metrics:
                cost_history[metric.name][epoch] = metric.epoch_stat / num_batches

        return cost_history

    def evaluate(self, X, y, batch_size=None):
        return self.fit(X, y, batch_size=batch_size, train=False)

# A multilayer perception with specified filter count.
class MLP(Sequential):
    def __init__(self, input_neurons: int, layer_neurons: List[int], output_act=activations.Linear, intermediate_act=activations.SReLU):
        super().__init__()

        self.intermediate_act = intermediate_act
        self.output_act = output_act
        self.input_neurons = input_neurons
        self.layer_neurons = layer_neurons
        self._setup_layers()
        self.layers[-1].act_fn = output_act()

    def _setup_layers(self):
        for idx, neuron_count in enumerate(self.layer_neurons):
            self.add(layers.FC(neuron_count, act_fn=self.intermediate_act))

# A Sequential model with Conv2d layers with specified filter and stride count as well as an ending global pooling layer.
class CNN(Sequential):
    def __init__(self, input_filters: int, layer_filters: List[int], strides: List[int] = None, output_act=activations.Linear, intermediate_act=activations.SReLU):
        super().__init__()

        if strides is None:
            self.strides = [1 for _ in range(len(layer_filters))]
        else:
            self.strides = strides

        self.intermediate_act = intermediate_act
        self.output_act = output_act
        self.input_filters = input_filters
        self.layer_filters = layer_filters
        self._setup_layers()
        self.layers[-2].act_fn = output_act()

    def _setup_layers(self):
        for idx, filter_count in enumerate(self.layer_filters):
            self.add(layers.Conv2d(
                filter_count,
                act_fn=self.intermediate_act,
                stride=self.strides[idx]
            ))

        self.add(layers.GlobalPooling2d())
