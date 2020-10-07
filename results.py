from brian2 import *
import dill
import numpy as np
import os


def bins_to_centers(bins):
    return (bins[:-1] + bins[1:]) / 2


class Results(object):

    # inputs should be able to be gained only from params and results objects (arrays)

    def __init__(self, resultsIdentifier, loadFolder):
        self.rID = resultsIdentifier
        self.loadFolder = loadFolder
        self.load_params()
        self.load_results()

    def load_params(self):
        loadPath = os.path.join(self.loadFolder, self.rID + '_params.pkl')
        with open(loadPath, 'rb') as f:
            params = dill.load(f)
        self.p = params

    def load_results(self):
        loadPath = os.path.join(self.loadFolder, self.rID + '_results.npz')
        npzObject = np.load(loadPath)
        self.npzObject = npzObject

        self.spikeMonExcT = npzObject['spikeMonExcT']
        self.spikeMonExcI = npzObject['spikeMonExcI']
        self.spikeMonInhT = npzObject['spikeMonInhT']
        self.spikeMonInhI = npzObject['spikeMonInhI']
        self.stateMonExcV = npzObject['stateMonExcV']
        self.stateMonInhV = npzObject['stateMonInhV']

    def calculate_spike_rate(self):
        dtHist = float(5 * ms)
        histBins = arange(0, float(self.p['duration']), dtHist)
        histCenters = arange(0 + dtHist / 2, float(self.p['duration']) - dtHist / 2, dtHist)

        FRExc, _ = histogram(self.spikeMonExcT, histBins)
        FRInh, _ = histogram(self.spikeMonInhT, histBins)

        FRExc = FRExc / dtHist / self.p['nExc']
        FRInh = FRInh / dtHist / self.p['nInh']
        
        self.dtHist = dtHist
        self.histBins = histBins
        self.histCenters = histCenters
        self.FRExc = FRExc
        self.FRInh = FRInh

    def calculate_voltage_histogram(self):
        voltageNumpyExc = self.stateMonExcV[0, :]
        voltageNumpyInh = self.stateMonInhV[0, :]

        voltageHistBins = arange(voltageNumpyExc.min(), voltageNumpyExc.max(), .1)
        voltageHistCenters = bins_to_centers(voltageHistBins)

        voltageHistExc, _ = histogram(voltageNumpyExc, voltageHistBins)
        voltageHistInh, _ = histogram(voltageNumpyInh, voltageHistBins)
        
        self.voltageHistBins = voltageHistBins
        self.voltageHistCenters = voltageHistCenters
        self.voltageHistExc = voltageHistExc
        self.voltageHistInh = voltageHistInh

    def plot_spike_raster(self, ax):
        ax.scatter(self.spikeMonExcT, self.spikeMonExcI, c='g', s=1)
        ax.scatter(self.spikeMonInhT, self.p['nExcSpikemon'] + self.spikeMonInhI, c='r', s=1)
        ax.set(xlim=(0., self.p['duration'] / second), ylim=(0, self.p['nUnits']), ylabel='neuron index')

    def plot_firing_rate(self, ax):
        ax.plot(self.histCenters[:self.FRExc.size], self.FRExc, label='Exc', color='green', alpha=0.5)
        ax.plot(self.histCenters[:self.FRInh.size], self.FRInh, label='Inh', color='red', alpha=0.5)
        ax.legend()
        ax.set(xlabel='Time (s)', ylabel='Firing Rate (Hz)')

    def plot_voltage_histogram(self, ax):
        ax.plot(self.voltageHistCenters, self.voltageHistExc, color='green', alpha=0.5)
        ax.plot(self.voltageHistCenters, self.voltageHistInh, color='red', alpha=0.5)
        ax.vlines(self.p['eLeakExc'] / mV, 0, self.voltageHistExc.max() / 2, color='green', ls='--', alpha=0.5)
        ax.vlines(self.p['eLeakInh'] / mV, 0, self.voltageHistInh.max() / 2, color='red', ls='--', alpha=0.5)
        ax.set(xlabel='voltage (mV)', ylabel='# of occurences', xlim=(0, 20))
        # plt.yscale('log')

    def plot_voltage_detail(self, ax, unitType='Exc', useStateInd=0):
        # reconstruct the time vector
        stateMonT = np.arange(0, float(self.p['duration']), float(self.p['dt']))
        yLims = (self.p['eLeakExc'] / mV - 30, 30)

        if unitType == 'Exc':
            voltageSeries = self.stateMonExcV[useStateInd, :]
            spikeMonT = self.spikeMonExcT
            spikeMonI = self.spikeMonExcI
            useColor = 'green'
        elif unitType == 'Inh':
            voltageSeries = self.stateMonInhV[useStateInd, :]
            spikeMonT = self.spikeMonInhT
            spikeMonI = self.spikeMonInhI
            useColor = 'red'
        else:
            print('you messed up')
            return

        ax.axhline(self.p['vThresh' + unitType] / mV, color=useColor, linestyle=':')  # Threshold
        ax.axhline(self.p['eLeak' + unitType] / mV, color=useColor, linestyle='--')  # Resting
        ax.plot(stateMonT, voltageSeries, color=useColor, lw=.3)
        ax.vlines(spikeMonT[spikeMonI == useStateInd], 20, 60, color=useColor, lw=.3)
        ax.set(xlim=(0., self.p['duration'] / second), ylim=yLims, ylabel='mV')
