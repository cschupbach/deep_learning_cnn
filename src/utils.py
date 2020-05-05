import numpy as np
import torch
import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.gridspec as pltgs


class Load():
    def __init__(self, method='torch'):
        self.method = method

    def _channels_first(self, x, rows, cols):
        return x.reshape(x.shape[0], 1, rows, cols)

    def _channels_last(self, x, rows, cols):
        return x.reshape(x.shape[0], rows, cols, 1)

    def _reshape(self, x_train, x_test):
        rows, cols = x_train.shape[-2:]
        if K.image_data_format() == 'channels_first' or self.method == 'torch':
            x_train = self._channels_first(x_train, rows, cols)
            x_test = self._channels_first(x_test, rows, cols)
            return x_train, x_test, (1, rows, cols)
        else:
            x_train = self._channels_last(x_train, rows, cols)
            x_test = self._channels_last(x_test, rows, cols)
            return x_train, x_test, (rows, cols, 1)

    def _normalize(self, x):
        return x.astype('float32') / 255

    def _numpy_to_tensor(self, x):
        if self.method == 'torch':
            return torch.Tensor(x)
        else:
            return tf.convert_to_tensor(x)

    def mnist(self, sample=None):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test, input_shape = self._reshape(x_train, x_test)
        x_train = self._numpy_to_tensor(self._normalize(x_train))
        if type(sample) == int:
            print(x_train[sample:sample+1].shape)
            return x_train[sample:sample+1]
        else:
            x_train = self._numpy_to_tensor(self._normalize(x_train))
            x_test = self._numpy_to_tensor(self._normalize(x_test))
            y_train = to_categorical(y_train, len(np.unique(y_train)))
            y_train = self._numpy_to_tensor(y_train)
            y_test = to_categorical(y_test, len(np.unique(y_test)))
            y_test = self._numpy_to_tensor(y_test)
            return x_train, y_train, x_test, y_test, input_shape

    def image(self, fn):
        x = plt.imread(fn).astype('float32') / 255
        H, W = x.shape[:2]
        if x.ndim == 2:
            x = x.reshape(1, 1, H, W)
            return self._numpy_to_tensor(x)
        else:
            x = np.transpose(x, [2, 0, 1])
            x = x.reshape(-1, 1, H, W)
        return self._numpy_to_tensor(x)


class Plot():
    def __init__(self, width=10, height=4, cmap='magma', size=12, bot=0.1):
        self.width = width
        self.height = height
        self.cmap = cmap
        self.size = size
        self.fig = plt.figure(figsize=(self.width, self.height))
        self.bot = bot

    def _no_ticks(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        return None

    def _get_lims(self, ax):
        return np.array(ax.get_xlim()), np.array(ax.get_ylim())

    def _shape_label(self, x, xlim, ylim):
        _shape = tuple([x.shape[i] for i in [1, 2, 0]])
        plt.text(np.mean(xlim), np.max(ylim)*1.05, _shape,
                 ha='center', va='top', size=self.size)
        return None

    def _level_locs(self, levels):
        x1 = [1 / (levels * 4)]
        x1 += [(i + 0.8) / levels for i in range(1, levels-1)]
        x2 = [(i + 0.7) / levels for i in range(1, levels-1)]
        x2 += [(levels - 0.25) / levels]
        return np.array(x1), np.array(x2)


    def _network_desc(self, levels):
        x1, x2 = self._level_locs(levels)
        gs = pltgs.GridSpec(1, 1, left=0, right=1, top=self.bot, bottom=0)
        ax = self.fig.add_subplot(gs[0,0])
        for i in range(len(x1)):
            ax.plot([x1[i], x1[i]], [0.9, 0.7], c='k')
            ax.plot([x1[i], x2[i]], [0.7, 0.7], c='k')
            ax.plot([x2[i], x2[i]], [0.9, 0.7], c='k')
            ax.text((x1[i] + x2[i]) / 2, 0.55, 'Convolution + ReLU',
                     ha='center', va='top', size=self.size)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.axis('off')
        return None

    def _conv_desc(self, levels, activation):
        x1, x2 = self._level_locs(levels)
        label = 'Convolution'
        if len(activation) > 0:
            label += f'\n+ {activation}'
        gs = pltgs.GridSpec(1, 1, left=0, right=1, top=1, bottom=0)
        ax = self.fig.add_subplot(gs[0,0])
        for i in range(len(x1)):
            ax.plot([0.27, 0.45], [0.50, 0.50], c='k')
            ax.plot([0.45, 0.44], [0.50, 0.52], c='k')
            ax.plot([0.45, 0.44], [0.50, 0.48], c='k')
            ax.text((0.27 + 0.45) / 2, 0.47, label, ha='center', va='top',
                    size=self.size)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.axis('off')
        return None

    def _plot_input(self, x_input, levels):
        gs = pltgs.GridSpec(1, 2, left=0, right=1/levels, top=1,
                            bottom=self.bot)
        ax = self.fig.add_subplot(gs[0,0])
        ax.imshow(x_input[0,0], cmap=self.cmap, aspect='equal')
        xlim, ylim = self._get_lims(ax)
        plt.text(np.mean(xlim), -np.max(ylim)*0.05, 'Input',
                 ha='center', va='bottom', size=self.size)
        self._shape_label(x_input[0], xlim, ylim)
        self._no_ticks(ax)
        return None

    def _network_layout(self, x_input, x_list):
        input_size = x_input[0,0].shape[0]
        xrng = np.arange(len(x_list))
        x = [x_list[i][0,:,:,:] for i in xrng]
        layers = [x[i].shape[0] for i in xrng]
        size_ratio = [x[i].shape[1] / input_size for i in xrng]
        ws = [(1 / ((layers[i] - 1) * size_ratio[i])) - 1 for i in xrng]
        hs = [(1 / ((layers[i] - 1) * size_ratio[i])) - 1 for i in xrng]
        return x, xrng, layers, ws, hs

    def _plot_network(self, x_input, x_list, levels):
        x, xrng, layers, ws, hs = self._network_layout(x_input, x_list)
        for i in xrng:
            gs = pltgs.GridSpec(layers[i], layers[i], left=(i+1)/levels,
                                right=(i+2)/levels, top=1, bottom=self.bot,
                                wspace=ws[i], hspace=hs[i])
            for j in range(layers[i]):
                ax = self.fig.add_subplot(gs[j,j])
                ax.imshow(x[i][j], cmap=self.cmap, aspect='equal')
                self._no_ticks(ax)
            xlim, ylim = self._get_lims(ax)
            self._shape_label(x[i], xlim, ylim)
        return None

    def network(self, x_input, x_list, activation='', channels='first'):
        if channels == 'last':
            x_input = np.transpose(x_input, [0, 3, 1, 2])
            x_list = [np.transpose(x, [0, 3, 1, 2]) for x in x_list]
        levels = len(x_list) + 1
        if levels > 2:
            self._network_desc(levels)
        else:
            self.bot = 0
            self._conv_desc(levels, activation)
        self._plot_input(x_input, levels)
        self._plot_network(x_input, x_list, levels)
        return None



























#
#
