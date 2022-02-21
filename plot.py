"""
dedicated plotting functionality that does not belong to a specific class
"""

from brian2 import ms, mV, second
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


def spike_raster(params, spikeMonExcT, spikeMonInhT, spikeMonExcI, spikeMonInhI):
    fig1 = plt.figure(num=1, figsize=(18, 10))
    plt.clf()
    fig1, ax = plt.subplots(num=1)
    ax.scatter(spikeMonExcT / ms, spikeMonExcI, c='g', s=1,
               marker='.', linewidths=0)
    ax.scatter(spikeMonInhT / ms, params['nExcSpikemon'] + spikeMonInhI, c='r', s=1,
               marker='.', linewidths=0)
    ax.set(xlim=(0., params['duration'] / ms), ylim=(0, params['nUnits']), ylabel='neuron index')
    return fig1


def firing_rate(histCenters, FRExc, FRInh):
    fig3 = plt.figure(num=3, figsize=(18, 4))
    plt.clf()
    fig3, ax = plt.subplots(1, 1, sharex=True, num=3)
    ax.plot(histCenters[:FRExc.size], FRExc, label='Exc', color='green', alpha=0.5)
    ax.plot(histCenters[:FRInh.size], FRInh, label='Inh', color='red', alpha=0.5)
    ax.legend()
    ax.set(xlabel='Time (s)', ylabel='Firing Rate (Hz)')
    return fig3


def voltage_histogram(params, voltageHistCenters, voltageHistExc, voltageHistInh):
    fig4 = plt.figure(num=4, figsize=(10, 5))
    plt.clf()
    fig4, ax = plt.subplots(1, 1, num=4)
    ax.plot(voltageHistCenters, voltageHistExc, color='green', alpha=0.5)
    ax.plot(voltageHistCenters, voltageHistInh, color='red', alpha=0.5)
    ax.vlines(params['eLeakExc'] / mV, 0, voltageHistExc.max() / 2, color='green', ls='--', alpha=0.5)
    ax.vlines(params['eLeakInh'] / mV, 0, voltageHistInh.max() / 2, color='red', ls='--', alpha=0.5)
    ax.set(xlabel='voltage (mV)', ylabel='# of occurences', xlim=(0, 20))
    # plt.yscale('log')
    return fig4


def voltage_detail(params, stateMonExcT, stateMonInhT, stateMonExcV, stateMonInhV,
                   spikeMonExcT, spikeMonInhT, spikeMonExcI, spikeMonInhI):
    yLims = (params['eLeakExc'] / mV - 30, 30)

    fig5 = plt.figure(num=5, figsize=(18, 6))
    plt.clf()
    fig5, ax = plt.subplots(2, 1, sharex=True, sharey=True, num=5)

    ax[0].axhline(params['vThreshExc'] / mV, color='g', linestyle=':')  # Threshold
    ax[0].axhline(params['eLeakExc'] / mV, color='g', linestyle='--')  # Resting
    ax[0].plot(stateMonExcT / ms, stateMonExcV[0, :] / mV, color='green', lw=.3)
    ax[0].vlines(spikeMonExcT[spikeMonExcI == params['stateIndExc']] / ms, 20, 60, color='green', lw=.3)
    ax[0].set(xlim=(0., params['duration'] / ms), ylim=yLims, ylabel='mV')

    ax[1].axhline(params['vThreshInh'] / mV, color='r', linestyle=':')  # Threshold
    ax[1].axhline(params['eLeakInh'] / mV, color='r', linestyle='--')  # Threshold
    ax[1].plot(stateMonInhT / ms, stateMonInhV[0, :] / mV, color='red', lw=.3)
    ax[1].vlines(spikeMonInhT[spikeMonInhI == params['stateIndInh']] / ms, 20, 60, color='red', lw=.3)
    ax[1].set(xlim=(0., params['duration'] / ms), ylim=yLims, ylabel='mV', xlabel='Time (ms)')

    return fig5


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
        ytl.set_fontproperties(fontProp)
    cb.outline.set_visible((not removeFrame))


def remove_axes_less(axs):
    rows, cols = axs.shape
    for ri in range(rows - 2):
        for ci in range(cols):
            axs[ri][ci].axis('off')


def prune_figure_less(axs):
    """ prune unnecessary labels """
    rows, cols = axs.shape
    # prune unnecessary xlabels and ylabels
    for ri in range(rows - 2):
        for ci in range(cols):
            # axs[ri][ci].set_xlabel('')
            axs[ri][ci].get_xaxis().set_label_text('')
            axs[ri][ci].get_xaxis().set_ticks([])
    for ri in range(rows):
        for ci in range(1, cols):
            axs[ri][ci].get_yaxis().set_label_text('')
            axs[ri][ci].get_yaxis().set_ticks([])


def prune_figure(axs, pruneYTicks=True):
    """ prune unnecessary labels """
    rows, cols = axs.shape
    # prune unnecessary xlabels and ylabels
    for ri in range(rows - 1):
        for ci in range(cols):
            # axs[ri][ci].set_xlabel('')
            axs[ri][ci].get_xaxis().set_label_text('')
            axs[ri][ci].get_xaxis().set_ticks([])
    for ri in range(rows):
        for ci in range(1, cols):
            axs[ri][ci].get_yaxis().set_label_text('')
            if pruneYTicks:
                axs[ri][ci].get_yaxis().set_ticks([])


def prune_figure_more(axs, pruneYTicks=True):
    """ prune unnecessary labels """
    shape = axs.shape
    if len(shape) == 2:
        rows, cols = shape
        # prune unnecessary xlabels and ylabels
        for ri in range(rows):
            for ci in range(cols):
                # axs[ri][ci].set_xlabel('')
                axs[ri][ci].get_xaxis().set_label_text('')
                axs[ri][ci].get_xaxis().set_ticks([])
                axs[ri][ci].get_yaxis().set_label_text('')
                if pruneYTicks:
                    axs[ri][ci].get_yaxis().set_ticks([])
    elif len(shape) == 1:
        for ax in axs:
            ax.get_xaxis().set_label_text('')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_label_text('')
            if pruneYTicks:
                ax.get_yaxis().set_ticks([])


def plot_spike_raster(ax, p, spikeMonExcI, spikeMonInhI, spikeMonExcT, spikeMonInhT, downSampleUnits=True, rng=None):

    if not rng:
        rng = np.random.default_rng(None)  # random seed

    if downSampleUnits:
        targetDisplayedExcUnits = 100
        targetDisplayedInhUnits = 100
        downSampleE = rng.choice(p['nExc'], size=targetDisplayedExcUnits, replace=False)
        downSampleI = rng.choice(p['nInh'], size=targetDisplayedInhUnits, replace=False)
        matchingEUnitsBool = np.isin(spikeMonExcI, downSampleE)
        matchingIUnitsBool = np.isin(spikeMonInhI, downSampleI)
        DownSampleERev = np.full((downSampleE.max() + 1,), np.nan)
        DownSampleERev[downSampleE] = np.arange(downSampleE.size)
        DownSampleIRev = np.full((downSampleI.max() + 1,), np.nan)
        DownSampleIRev[downSampleI] = np.arange(downSampleI.size)
        xExc = spikeMonExcT[matchingEUnitsBool]
        yExc = DownSampleERev[spikeMonExcI[matchingEUnitsBool].astype(int)]
        xInh = spikeMonInhT[matchingIUnitsBool]
        yInh = targetDisplayedExcUnits + DownSampleIRev[spikeMonInhI[matchingIUnitsBool].astype(int)]
        yLims = (0, targetDisplayedExcUnits + targetDisplayedInhUnits)
    else:
        xExc = spikeMonExcT
        yExc = spikeMonExcI
        xInh = spikeMonInhT
        yInh = p['nExc'] + spikeMonInhI

    ax.scatter(xExc, yExc, c='g', s=1, marker='.', linewidths=0)
    ax.scatter(xInh, yInh, c='r', s=1, marker='.', linewidths=0)
    ax.set(xlim=(0., p['duration'] / second), ylim=yLims, ylabel='neuron index')


def plot_firing_rate(ax, histCenters, FRExc, FRInh):
    ax.plot(histCenters[:FRExc.size], FRExc, label='Exc', color='green', alpha=0.5)
    ax.plot(histCenters[:FRInh.size], FRInh, label='Inh', color='red', alpha=0.5)
    # ax.legend()
    ax.set(ylabel='Firing Rate (Hz)')


def plot_voltage_detail(ax, p, stateMonT, voltageSeries, yLims, unitType='Exc', **kwargs):

    if unitType == 'Exc':
        useColor = 'green'
    elif unitType == 'Inh':
        useColor = 'red'
    else:
        print('you messed up')
        return

    ax.plot(stateMonT, voltageSeries, color=useColor, lw=.3, **kwargs)
    ax.set(xlim=(0., p['duration'] / second), ylim=yLims, ylabel='mV')
