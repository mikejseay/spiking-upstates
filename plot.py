"""
dedicated plotting functionality that does not belong to a specific class
"""

from brian2 import *


def spike_raster(params, spikeMonExcT, spikeMonInhT, spikeMonExcI, spikeMonInhI):
    fig1 = figure(num=1, figsize=(18, 10))
    clf()
    fig1, ax = plt.subplots(num=1)
    ax.scatter(spikeMonExcT / ms, spikeMonExcI, c='g', s=1,
               marker='.', linewidths=0)
    ax.scatter(spikeMonInhT / ms, params['nExcSpikemon'] + spikeMonInhI, c='r', s=1,
               marker='.', linewidths=0)
    ax.set(xlim=(0., params['duration'] / ms), ylim=(0, params['nUnits']), ylabel='neuron index')
    return fig1


def firing_rate(histCenters, FRExc, FRInh):
    fig3 = figure(num=3, figsize=(18, 4))
    clf()
    fig3, ax = plt.subplots(1, 1, sharex=True, num=3)
    ax.plot(histCenters[:FRExc.size], FRExc, label='Exc', color='green', alpha=0.5)
    ax.plot(histCenters[:FRInh.size], FRInh, label='Inh', color='red', alpha=0.5)
    ax.legend()
    ax.set(xlabel='Time (s)', ylabel='Firing Rate (Hz)')
    return fig3


def voltage_histogram(params, voltageHistCenters, voltageHistExc, voltageHistInh):
    fig4 = figure(num=4, figsize=(10, 5))
    clf()
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

    fig5 = figure(num=5, figsize=(18, 6))
    clf()
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
