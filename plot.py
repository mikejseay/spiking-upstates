import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


class MidpointNormalize(colors.Normalize):
    ''' create asymmetric norm '''

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def weight_matrix(ax, values,
                  useCmap='RdBu_r', limsMethod='absmax',
                  xlabel='', ylabel='', clabel='',
                  vlims=None, fontproperties=None, removeFrame=True):
    """ given an axis handle, an array of values, and some optional params,
        visualize a weight matrix in a heat map using imshow
    """

    i = ax.imshow(values,
                  cmap=getattr(plt.cm, useCmap),
                  aspect='auto',
                  interpolation='none')
    ax.set_xlabel(xlabel, fontproperties=fontproperties)
    ax.set_ylabel(ylabel, fontproperties=fontproperties)
    for xtl in ax.get_xticklabels():
        xtl.set_fontproperties(fontproperties)
    for ytl in ax.get_yticklabels():
        ytl.set_fontproperties(fontproperties)
    if removeFrame:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    if limsMethod == 'absmax':
        vmax = np.nanmax(np.fabs(values))
        vmin = -vmax
    elif limsMethod == 'minmax':
        vmax, vmin = np.nanmax(values), np.nanmin(values)
    elif limsMethod == 'custom':
        vmin, vmax = vlims

    if vmin != -vmax and vmin < 0 and vmax > 0:
        norm = MidpointNormalize(vmin, vmax, 0)
    else:
        norm = False

    i.set_clim(vmin, vmax)
    if norm:
        i.set_norm(norm)

    cb = plt.colorbar(i, ax=ax)
    cb.ax.set_ylabel(clabel, rotation=270, fontproperties=fontproperties, labelpad=20)
    for ytl in cb.ax.get_yticklabels():
        ytl.set_fontproperties(fontproperties)
    cb.outline.set_visible((not removeFrame))