import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as pltgs


def _network_dict(network_id):
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


def _no_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    return None


def _get_lims(ax):
    return np.array(ax.get_xlim()), np.array(ax.get_ylim())


def _shape_label(x, xlim, ylim, size):
    _shape = tuple([x.shape[i] for i in [1, 2, 0]])
    plt.text(np.mean(xlim), np.max(ylim)*1.05, _shape,
             ha='center', va='top', size=size)
    return None


def _level_locs(levels):
    x1 = [1 / (levels * 4)]
    x1 += [(i + 0.8) / levels for i in range(1, levels-1)]
    x2 = [(i + 0.7) / levels for i in range(1, levels-1)]
    x2 += [(levels - 0.25) / levels]
    return np.array(x1), np.array(x2)


def _network_desc(fig, levels, network_id, bot, size):
    x1, x2 = _level_locs(levels)
    labels = _network_dict(network_id)
    gs = pltgs.GridSpec(1, 1, left=0, right=1, top=bot, bottom=0)
    ax = fig.add_subplot(gs[0,0])
    for i in range(len(x1)):
        ax.plot([x1[i], x1[i]], [0.8, 0.7], c='k')
        ax.plot([x1[i], x2[i]], [0.7, 0.7], c='k')
        ax.plot([x2[i], x2[i]], [0.8, 0.7], c='k')
        ax.text((x1[i] + x2[i]) / 2, 0.55, labels[i],
                 ha='center', va='top', size=size)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.axis('off')
    return None


def _conv_desc(fig, levels, activation, size):
    x1, x2 = _level_locs(levels)
    label = 'Convolution'
    if len(activation) > 0:
        label += f'\n+ {activation}'
    gs = pltgs.GridSpec(1, 1, left=0, right=1, top=1, bottom=0)
    ax = fig.add_subplot(gs[0,0])
    for i in range(len(x1)):
        ax.plot([0.27, 0.45], [0.50, 0.50], c='k')
        ax.plot([0.45, 0.44], [0.50, 0.52], c='k')
        ax.plot([0.45, 0.44], [0.50, 0.48], c='k')
        ax.text((0.27 + 0.45) / 2, 0.47, label, ha='center', va='top',
                size=size)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.axis('off')
    return None


def _plot_input(fig, x_input, levels, bot, cmap, size):
    gs = pltgs.GridSpec(1, 2, left=0, right=1/levels, top=1,
                        bottom=bot)
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(x_input[0,0], cmap=cmap, aspect='equal')
    xlim, ylim = _get_lims(ax)
    plt.text(np.mean(xlim), -np.max(ylim)*0.05, 'Input',
             ha='center', va='bottom', size=size)
    _shape_label(x_input[0], xlim, ylim, size)
    _no_ticks(ax)
    return None


def _network_layout(x_input, x_list):
    input_size = x_input[0,0].shape[0]
    xrng = np.arange(len(x_list))
    x = [x_list[i][0,:,:,:] for i in xrng]
    layers = [x[i].shape[0] for i in xrng]
    size_ratio = [x[i].shape[1] / input_size for i in xrng]
    ws = [(1 / ((layers[i] - 1) * size_ratio[i])) - 1 for i in xrng]
    hs = [(1 / ((layers[i] - 1) * size_ratio[i])) - 1 for i in xrng]
    return x, xrng, layers, ws, hs


def _plot_network(fig, x_input, x_list, levels, bot, cmap, size):
    x, xrng, layers, ws, hs = _network_layout(x_input, x_list)
    for i in xrng:
        gs = pltgs.GridSpec(layers[i], layers[i], left=(i+1)/levels,
                            right=(i+2)/levels, top=1, bottom=bot,
                            wspace=ws[i], hspace=hs[i])
        for j in range(layers[i]):
            ax = fig.add_subplot(gs[j,j])
            ax.imshow(x[i][j], cmap=cmap, aspect='equal')
            _no_ticks(ax)
        xlim, ylim = _get_lims(ax)
        _shape_label(x[i], xlim, ylim, size)
    return None


def network(x_input, x_list, activation='', network_id=0, channels='first',
            height=4, width=10, cmap='magma', size=12, bot=0.1, dpi=200,
            save_as=None):
    fig = plt.figure(figsize=(width, height))
    if channels == 'last':
        x_input = np.transpose(x_input, [0, 3, 1, 2])
        x_list = [np.transpose(x, [0, 3, 1, 2]) for x in x_list]
    levels = len(x_list) + 1
    if levels > 2:
        _network_desc(fig, levels, network_id, bot, size)
    else:
        bot = 0
        _conv_desc(fig, levels, activation, size)
    _plot_input(fig, x_input, levels, bot, cmap, size)
    _plot_network(fig, x_input, x_list, levels, bot, cmap, size)
    if save_as:
        plt.close()
        fig.savefig(f'fig/figure{save_as}.png', bbox_inches='tight', dpi=dpi)
    return None


# class Plot():
#     def __init__(self, width=10, height=4, cmap='magma', size=12, bot=0.1):
#         self.width = width
#         self.height = height
#         self.cmap = cmap
#         self.size = size
#         self.fig = plt.figure(figsize=(self.width, self.height))
#         self.bot = bot
#
#     def _network_dict(self, network_id):
#         c = 'Convolution\n+ '
#         dictionary = {
#             0: [c + 'ReLU', r'Max Pooling $(2 \times 2)$'],
#             1: [c + 'Tanh', r'Max Pooling $(2 \times 2)$'],
#             2: [c + 'ReLU', r'Average Pooling $(2 \times 2)$'],
#             3: [c + 'Tanh', r'Average Pooling $(2 \times 2)$'],
#             4: [c + 'Tanh', r'Max Pooling $(2 \times 2)$', c + 'ReLU'],
#             5: [c + 'ReLU', r'Max Pooling $(2 \times 2)$', c + 'ReLU',
#                 c + 'ReLU', r'Max Pooling $(2 \times 2)$']
#         }
#         return dictionary[network_id]
#
#     def _no_ticks(self, ax):
#         ax.set_xticks([])
#         ax.set_yticks([])
#         return None
#
#     def _get_lims(self, ax):
#         return np.array(ax.get_xlim()), np.array(ax.get_ylim())
#
#     def _shape_label(self, x, xlim, ylim):
#         _shape = tuple([x.shape[i] for i in [1, 2, 0]])
#         plt.text(np.mean(xlim), np.max(ylim)*1.05, _shape,
#                  ha='center', va='top', size=self.size)
#         return None
#
#     def _level_locs(self, levels):
#         x1 = [1 / (levels * 4)]
#         x1 += [(i + 0.8) / levels for i in range(1, levels-1)]
#         x2 = [(i + 0.7) / levels for i in range(1, levels-1)]
#         x2 += [(levels - 0.25) / levels]
#         return np.array(x1), np.array(x2)
#
#
#     def _network_desc(self, levels, network_id):
#         x1, x2 = self._level_locs(levels)
#         labels = self._network_dict(network_id)
#         gs = pltgs.GridSpec(1, 1, left=0, right=1, top=self.bot, bottom=0)
#         ax = self.fig.add_subplot(gs[0,0])
#         for i in range(len(x1)):
#             ax.plot([x1[i], x1[i]], [0.8, 0.7], c='k')
#             ax.plot([x1[i], x2[i]], [0.7, 0.7], c='k')
#             ax.plot([x2[i], x2[i]], [0.8, 0.7], c='k')
#             ax.text((x1[i] + x2[i]) / 2, 0.55, labels[i],
#                      ha='center', va='top', size=self.size)
#         ax.set_ylim(0, 1)
#         ax.set_xlim(0, 1)
#         ax.axis('off')
#         return None
#
#     def _conv_desc(self, levels, activation):
#         x1, x2 = self._level_locs(levels)
#         label = 'Convolution'
#         if len(activation) > 0:
#             label += f'\n+ {activation}'
#         gs = pltgs.GridSpec(1, 1, left=0, right=1, top=1, bottom=0)
#         ax = self.fig.add_subplot(gs[0,0])
#         for i in range(len(x1)):
#             ax.plot([0.27, 0.45], [0.50, 0.50], c='k')
#             ax.plot([0.45, 0.44], [0.50, 0.52], c='k')
#             ax.plot([0.45, 0.44], [0.50, 0.48], c='k')
#             ax.text((0.27 + 0.45) / 2, 0.47, label, ha='center', va='top',
#                     size=self.size)
#         ax.set_ylim(0, 1)
#         ax.set_xlim(0, 1)
#         ax.axis('off')
#         return None
#
#     def _plot_input(self, x_input, levels):
#         gs = pltgs.GridSpec(1, 2, left=0, right=1/levels, top=1,
#                             bottom=self.bot)
#         ax = self.fig.add_subplot(gs[0,0])
#         ax.imshow(x_input[0,0], cmap=self.cmap, aspect='equal')
#         xlim, ylim = self._get_lims(ax)
#         plt.text(np.mean(xlim), -np.max(ylim)*0.05, 'Input',
#                  ha='center', va='bottom', size=self.size)
#         self._shape_label(x_input[0], xlim, ylim)
#         self._no_ticks(ax)
#         return None
#
#     def _network_layout(self, x_input, x_list):
#         input_size = x_input[0,0].shape[0]
#         xrng = np.arange(len(x_list))
#         x = [x_list[i][0,:,:,:] for i in xrng]
#         layers = [x[i].shape[0] for i in xrng]
#         size_ratio = [x[i].shape[1] / input_size for i in xrng]
#         ws = [(1 / ((layers[i] - 1) * size_ratio[i])) - 1 for i in xrng]
#         hs = [(1 / ((layers[i] - 1) * size_ratio[i])) - 1 for i in xrng]
#         return x, xrng, layers, ws, hs
#
#     def _plot_network(self, x_input, x_list, levels):
#         x, xrng, layers, ws, hs = self._network_layout(x_input, x_list)
#         for i in xrng:
#             gs = pltgs.GridSpec(layers[i], layers[i], left=(i+1)/levels,
#                                 right=(i+2)/levels, top=1, bottom=self.bot,
#                                 wspace=ws[i], hspace=hs[i])
#             for j in range(layers[i]):
#                 ax = self.fig.add_subplot(gs[j,j])
#                 ax.imshow(x[i][j], cmap=self.cmap, aspect='equal')
#                 self._no_ticks(ax)
#             xlim, ylim = self._get_lims(ax)
#             self._shape_label(x[i], xlim, ylim)
#         return None
#
#     def network(self, x_input, x_list, activation='', network_id=0,
#                 channels='first'):
#         if channels == 'last':
#             x_input = np.transpose(x_input, [0, 3, 1, 2])
#             x_list = [np.transpose(x, [0, 3, 1, 2]) for x in x_list]
#         levels = len(x_list) + 1
#         if levels > 2:
#             self._network_desc(levels, network_id)
#         else:
#             self.bot = 0
#             self._conv_desc(levels, activation)
#         self._plot_input(x_input, levels)
#         self._plot_network(x_input, x_list, levels)
#         return None
