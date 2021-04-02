"""
dedicated plotting functionality that does not belong to a specific class
"""

from brian2 import ms, mV
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
                  xlabel='', ylabel='', clabel=''):
    """ given an axis handle, an array of values, and some optional params,
        visualize a weight matrix in a heat map using imshow
    """

    i = ax.imshow(values,
                  cmap=getattr(plt.cm, useCmap),
                  aspect='auto',
                  interpolation='none')
    ax.set(xlabel=xlabel, ylabel=ylabel)

    if limsMethod == 'absmax':
        vmax = np.max(np.fabs(values))
        vmin = -vmax
    elif limsMethod == 'minmax':
        vmax, vmin = np.max(values), np.min(values)

    if vmin != -vmax:
        norm = MidpointNormalize(vmin, vmax, 0)
    else:
        norm = False

    i.set_clim(vmin, vmax)
    if norm:
        i.set_norm(norm)

    cb = plt.colorbar(i, ax=ax)
    cb.ax.set_ylabel(clabel, rotation=270)

