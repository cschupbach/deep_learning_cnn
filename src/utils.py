import os

path = '/Users/chadschupbach/opt/anaconda3/'
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
os.environ['RUNFILES_DIR'] = path + 'share/plaidml'
os.environ['PLAIDML_NATIVE_PATH'] = path + 'lib/libplaidml.dylib'

import numpy as np
import pandas as pd
import torch
import keras
from keras import backend as K
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as pltgs


class Plot():
    def __init__(self, width=10, height=4, cmap='magma', size=12, bot=0.1):
        self.width = width
        self.height = height
        self.cmap = cmap
        self.size = size
        self.fig = plt.figure(figsize=(self.width, self.height))
        self.bot = bot

    def _network_dict(self, network_id):
        c = 'Convolution\n+ '
        dictionary = {
            0: [c + 'ReLU', r'Max Pooling $(2 \times 2)$'],
            1: [c + 'Tanh', r'Max Pooling $(2 \times 2)$'],
            2: [c + 'ReLU', r'Average Pooling $(2 \times 2)$'],
            3: [c + 'Tanh', r'Average Pooling $(2 \times 2)$'],
            4: [c + 'Tanh', r'Max Pooling $(2 \times 2)$', c + 'ReLU'],
            5: [c + 'ReLU', r'Max Pooling $(2 \times 2)$', c + 'ReLU',
                c + 'ReLU', r'Max Pooling $(2 \times 2)$']
        }
        return dictionary[network_id]

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


    def _network_desc(self, levels, network_id):
        x1, x2 = self._level_locs(levels)
        labels = self._network_dict(network_id)
        gs = pltgs.GridSpec(1, 1, left=0, right=1, top=self.bot, bottom=0)
        ax = self.fig.add_subplot(gs[0,0])
        for i in range(len(x1)):
            ax.plot([x1[i], x1[i]], [0.8, 0.7], c='k')
            ax.plot([x1[i], x2[i]], [0.7, 0.7], c='k')
            ax.plot([x2[i], x2[i]], [0.8, 0.7], c='k')
            ax.text((x1[i] + x2[i]) / 2, 0.55, labels[i],
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

    def network(self, x_input, x_list, activation='', network_id=0,
                channels='first'):
        if channels == 'last':
            x_input = np.transpose(x_input, [0, 3, 1, 2])
            x_list = [np.transpose(x, [0, 3, 1, 2]) for x in x_list]
        levels = len(x_list) + 1
        if levels > 2:
            self._network_desc(levels, network_id)
        else:
            self.bot = 0
            self._conv_desc(levels, activation)
        self._plot_input(x_input, levels)
        self._plot_network(x_input, x_list, levels)
        return None


def _channels_first(x, rows, cols, channels):
    x = x.reshape(x.shape[0], rows, cols, channels)
    return np.transpose(x, [0, 3, 1, 2])


def _channels_last(x, rows, cols, channels):
    return x.reshape(x.shape[0], rows, cols, channels)


def _reshape(x_train, x_test, method):
    rows, cols = x_train.shape[1:3]
    channels = x_train.shape[-1] if x_train.ndim > 3 else 1
    if method == 'keras':
        x_train = _channels_last(x_train, rows, cols, channels)
        x_test = _channels_last(x_test, rows, cols, channels)
        return x_train, x_test, (rows, cols, channels)
    else:
        x_train = _channels_first(x_train, rows, cols, channels)
        x_test = _channels_first(x_test, rows, cols, channels)
        return x_train, x_test, (channels, rows, cols)


def _normalize(x):
    return x.astype('float32') / 255


def _numpy_to_tensor(x, method):
    if method == 'keras':
        return K.constant(x)
    else:
        return torch.Tensor(x)


def dataset_classes(dataset='digits'):
    if dataset == 'fashion':
        return {0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
                4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag',
                9: 'Boot'}
    if dataset == 'cifar10':
        return {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    else:
        return {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}


def load_mnist(method='keras', dataset='digits', sample=None):
    if dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test, input_shape = _reshape(x_train, x_test, method)
    x_train = _numpy_to_tensor(_normalize(x_train), method)
    if type(sample) == int:
        print(x_train[sample:sample+1].shape)
        return x_train[sample:sample+1]
    else:
        x_test = _numpy_to_tensor(_normalize(x_test), method)
        y_train = to_categorical(y_train, len(np.unique(y_train)))
        y_train = _numpy_to_tensor(y_train, method)
        y_test = to_categorical(y_test, len(np.unique(y_test)))
        y_test = _numpy_to_tensor(y_test, method)
        if method == 'keras':
            x_train = K.eval(x_train)
            y_train = K.eval(y_train)
            x_test = K.eval(x_test)
            y_test = K.eval(y_test)
            return x_train, y_train, x_test, y_test, input_shape
        else:
            return x_train, y_train, x_test, y_test, input_shape


def load_image(fn, method='keras'):
    x = plt.imread(fn).astype('float32') / 255
    H, W = x.shape[:2]
    if x.ndim == 2:
        x = x.reshape(1, 1, H, W)
        return _numpy_to_tensor(x, method)
    else:
        x = np.transpose(x, [2, 0, 1])
        x = x.reshape(-1, 1, H, W)
    return _numpy_to_tensor(x, method)


def display_conv(x, conv, activation):
    x1 = K.eval(conv(x))
    Plot().network(K.eval(x), [x1], activation=activation, channels='last')
    return None


def display_conv_pool(x, conv_list, network_id, height=5):
    x_list = [conv_list[0](x)]
    for i in range(1, len(conv_list)):
        x_list += [conv_list[i](x_list[i-1])]
    x_list = [K.eval(x_list[i]) for i in range(len(conv_list))]
    Plot(height=height).network(K.eval(x), x_list, network_id=network_id,
                                channels='last')
    return None


def display_fc(model, x):
    label = model.get_layer(index=-1).name
    fc = K.eval(model(x))
    plt.figure(figsize=(10, 0.5))
    plt.imshow(fc, cmap='magma', aspect='auto')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    xloc = np.array(plt.gca().get_xlim())
    plt.text(xloc[0], -0.6, label, ha='left', va='bottom', size=12)
    plt.text(np.mean(xloc), 0.7, fc.shape, ha='center', va='top', size=12)
    return None


def get_labels(x_test, test_pred, test_act, test_res):
    idx = test_pred != test_act
    digit = x_test[idx][:,:,:,0]
    act_label = test_act[idx]
    pred_label = test_pred[idx]
    act_prob = test_res[idx, act_label]
    pred_prob = test_res[idx, pred_label]
    sidx = np.argsort(act_label)


def plot_samples(x_train, y_train, dataset='digits'):
    class_dict = dataset_classes(dataset)
    sample_list = [x_train[y_train[:,i].astype(bool)][:12] for i in range(10)]
    samples = np.concatenate(sample_list)
    gs = pltgs.GridSpec(10, 12, hspace=-0.025, wspace=-0.025)
    fig = plt.figure(figsize=(10, 8.5))
    yloc = np.linspace(0.95, 0.05, 10)
    if dataset != 'digits':
        for i in np.arange(10):
            plt.text(-0.01, yloc[i], class_dict[i], ha='right', va='center')
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
    for i in range(120):
        ax = fig.add_subplot(gs[i//12, i%12])
        ax.imshow(samples[i,:,:,0], cmap='magma')
        ax.set_xticks([])
        ax.set_yticks([])
    return None


def plot_class_means(x_train, y_train, dataset='digits'):
    class_dict = dataset_classes(dataset)
    means = []
    for i in range(y_train.shape[-1]):
        means += [x_train[(y_train[:,i] == 1)].mean(axis=0)[:,:,0]]
    gs = pltgs.GridSpec(2, 5)
    fig = plt.figure(figsize=(10, 4))
    for i in range(10):
        ax = fig.add_subplot(gs[i//5, i%5])
        ax.imshow(means[i], cmap='magma')
        ax.set_xticks([])
        ax.set_yticks([])
        xlim, ylim = np.array(ax.get_xlim()), np.array(ax.get_ylim())
        if dataset != 'digits':
            plt.text(np.mean(xlim), ylim[1] - np.ptp(ylim) * 0.02,
                     '{}'.format(class_dict[i]), ha='center',
                     va='bottom', size=12)
    return None


def training_summary(model, ledger, x_test, y_test, dataset='digits'):
    opt_epoch = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for i in range(len(model)):
        opt_epoch += [np.argmax(ledger[i].history['val_acc']) + 1]
        train_loss += [np.min(ledger[i].history['loss'])]
        train_acc += [np.max(ledger[i].history['acc'])]
        val_loss += [np.min(ledger[i].history['val_loss'])]
        val_acc += [np.max(ledger[i].history['val_acc'])]

    test_loss = []
    test_acc = []
    for i in range(len(model)):
        model[i].load_weights(f'models/{dataset}/best_weights_{i}.hdf5')
        score = model[i].evaluate(x_test, y_test, verbose=0)
        test_loss += [score[0]]
        test_acc += [score[1]]

    index = [f'Model {i}' for i in range(len(model))] + ['Average']
    df = pd.DataFrame({
        'Epoch': opt_epoch + [np.mean(opt_epoch)],
        'Train Loss': train_loss + [np.mean(train_loss)],
        'Val Loss': val_loss + [np.mean(val_loss)],
        'Test Loss': test_loss + [np.mean(test_loss)],
        'Train Acc': train_acc + [np.mean(train_acc)],
        'Val Acc': val_acc + [np.mean(val_acc)],
        'Test Acc': test_acc + [np.mean(test_acc)]
    }, index=index).round(5).round({
        'Train Acc': 4,
        'Val Acc': 4,
        'Test Acc': 4
    })
    return df


def ensemble_results(model, x_train, y_train, x_test, y_test):
    train_res = np.zeros(y_train.shape)
    test_res = np.zeros(y_test.shape)
    for i in range(len(model)):
        train_res += model[i].predict(x_train)
        test_res += model[i].predict(x_test)
    train_res /= len(model)
    test_res /= len(model)

    train_pred = np.argmax(train_res, axis=1)
    train_act = np.argmax(y_train, axis=1)
    ens_train_acc = (train_pred == train_act).mean()
    test_pred = np.argmax(test_res, axis=1)
    test_act = np.argmax(y_test, axis=1)
    ens_test_acc = (test_pred == test_act).mean()
    print('Ensemble Train Accuracy: {:.4f}'.format(ens_train_acc))
    print('Ensemble Test Accuracy: {:.4f}'.format(ens_test_acc))

    return test_act, test_pred, test_res


def plot_confusion(test_act, test_pred, dataset='digits'):
    title = 'MNIST CNN Ensemble\nConfusion Matrix'
    classes = list(dataset_classes(dataset).values())
    rotation = 90 if type(classes) == str else 0
    if dataset == 'fashion':
        title = dataset[0].upper() + dataset[1:] + ' ' + title
    if dataset == 'cifar10':
        title = dataset.upper() + title[5:]
    cmat = confusion_matrix(test_act, test_pred)
    misclass = np.max(cmat[np.abs(np.eye(len(classes)) - 1).astype(bool)])
    plt.figure(figsize=(8, 8))
    plt.imshow(cmat, cmap='PuBu', vmax=misclass*1.75)
    plt.title(title, size=13)
    plt.xticks(np.arange(10), classes, rotation=rotation)
    plt.yticks(np.arange(10), classes)
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            color = 'w' if cmat[i,j] > misclass else 'k'
            plt.text(j, i, cmat[i,j], ha='center', va='center',
                     color=color, size=12)
    plt.ylabel('Actual', size=12)
    plt.xlabel('Predicted', size=12)
    plt.tight_layout()
    return None


def plot_misclassified(x_test, test_pred, test_act, test_res, dataset='digits',
                       _sort=False, prelab='=', size=11, h_ratio=2.3):
    title = 'MNIST CNN Ensemble\nMisclassified Digits\n\n'
    prelab = '='
    class_dict = dataset_classes(dataset)
    if dataset == 'fashion':
        title = 'Fashion MNIST CNN Ensemble\nMisclassified Items (first 50)\n\n'
        prelab = '\n'
        size = 10
        h_ratio = 2.5
    if dataset == 'cifar10':
        title = dataset.upper() + title[5:]
    idx = test_pred != test_act
    digit = x_test[idx][:,:,:,0]
    act_label = test_act[idx]
    pred_label = test_pred[idx]
    act_prob = test_res[idx, act_label]
    pred_prob = test_res[idx, pred_label]
    if _sort:
        sorted_idx = np.argsort(act_label)
    else:
        sorted_idx = np.arange(len(act_label))

    n_misclass = len(sorted_idx)
    if n_misclass > 50:
        n_misclass = 50
        sorted_idx = sorted_idx[:50]
    nrows = int(np.ceil(n_misclass / 5))
    figsize = (10, nrows * h_ratio)
    gs = pltgs.GridSpec(nrows, 5)
    fig = plt.figure(figsize=figsize)
    plt.title(title, ha='center', va='center')
    plt.axis('off')
    for i, si in enumerate(sorted_idx):
        ax = fig.add_subplot(gs[i//5, i%5])
        ax.imshow(digit[si], cmap='magma')
        ax.set_xticks([])
        ax.set_yticks([])
        xlim, ylim = np.array(ax.get_xlim()), np.array(ax.get_ylim())
        plt.text(np.quantile(xlim, 0.25), ylim[1] - np.ptp(ylim) * 0.02,
                 'Actual{}{}\n{:.4f}'.format(prelab, class_dict[act_label[si]],
                 act_prob[si]), ha='center', va='bottom', size=size)
        plt.text(np.quantile(xlim, 0.75), ylim[1] - np.ptp(ylim) * 0.02,
                 'Pred{}{}\n{:.4f}'.format(prelab, class_dict[pred_label[si]],
                 pred_prob[si]), ha='center', va='bottom', size=size)
    return None
