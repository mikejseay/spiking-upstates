from brian2 import *
import dill
import numpy as np
import os
from functions import find_upstates
from stats import regress_linear
from scipy.stats import mode
from generate import convert_indices_times_to_dict


def bins_to_centers(bins):
    return (bins[:-1] + bins[1:]) / 2


def convert_sNMDA_to_current_exc_Jercog_CellNotSyn(JN, ind):
    v = JN.stateMonExc.v[ind, :]
    sE_NMDA = JN.stateMonExc.sE_NMDA[ind, :]
    hardGatePart = np.round(v > JN.p['vStepSigmoid'])
    NMDACurrent = JN.unitsExc.jE_NMDA[0] / (1 + exp(-JN.p['kSigmoid'] * (v - JN.p['vMidSigmoid']) / mV)) * sE_NMDA
    return hardGatePart * NMDACurrent


def convert_sNMDA_to_current_inh_Jercog_CellNotSyn(JN, ind):
    v = JN.stateMonInh.v[ind, :]
    sE_NMDA = JN.stateMonInh.sE_NMDA[ind, :]
    hardGatePart = np.round(v > JN.p['vStepSigmoid'])
    NMDACurrent = JN.unitsInh.jE_NMDA[0] / (1 + exp(-JN.p['kSigmoid'] * (v - JN.p['vMidSigmoid']) / mV)) * sE_NMDA
    return hardGatePart * NMDACurrent


def convert_sNMDA_to_current_exc_Jercog(JN, ind):
    v = JN.stateMonExc.v[ind, :]
    s_NMDA_tot = JN.stateMonExc.s_NMDA_tot[ind, :]
    hardGatePart = np.round(v > JN.p['vStepSigmoid'])
    NMDACurrent = JN.unitsExc.jE_NMDA[0] / (1 + exp(-JN.p['kSigmoid'] * (v - JN.p['vMidSigmoid']) / mV)) * s_NMDA_tot
    return hardGatePart * NMDACurrent


def convert_sNMDA_to_current_inh_Jercog(JN, ind):
    v = JN.stateMonInh.v[ind, :]
    s_NMDA_tot = JN.stateMonInh.s_NMDA_tot[ind, :]
    hardGatePart = np.round(v > JN.p['vStepSigmoid'])
    NMDACurrent = JN.unitsInh.jE_NMDA[0] / (1 + exp(-JN.p['kSigmoid'] * (v - JN.p['vMidSigmoid']) / mV)) * s_NMDA_tot
    return hardGatePart * NMDACurrent


def convert_sNMDA_to_current_exc_Destexhe(DN, ind):
    v = DN.stateMonExc.v[ind, :]
    s_NMDA_tot = DN.stateMonExc.s_NMDA_tot[ind, :]
    hardGatePart = np.round(v > DN.p['vStepSigmoid'])
    NMDACurrent = DN.unitsExc.ge_NMDA[0] / (1 + exp(-DN.p['kSigmoid'] * (v - DN.p['vMidSigmoid']) / mV)) * s_NMDA_tot
    return hardGatePart * NMDACurrent


def convert_sNMDA_to_current_inh_Destexhe(DN, ind):
    v = DN.stateMonInh.v[ind, :]
    s_NMDA_tot = DN.stateMonInh.s_NMDA_tot[ind, :]
    hardGatePart = np.round(v > DN.p['vStepSigmoid'])
    NMDACurrent = DN.unitsInh.ge_NMDA[0] / (1 + exp(-DN.p['kSigmoid'] * (v - DN.p['vMidSigmoid']) / mV)) * s_NMDA_tot
    return hardGatePart * NMDACurrent


class Results(object):

    # inputs should be able to be gained only from params and results objects (arrays)

    def __init__(self, resultsIdentifier, loadFolder):
        self.rID = resultsIdentifier
        self.loadFolder = loadFolder
        self.load_params()
        self.load_results()
        self.timeArray = np.arange(0, float(self.p['duration']), float(self.p['dt']))

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

        if 'spikeMonInpCorrT' in npzObject:
            self.spikeMonInpCorrT = npzObject['spikeMonInpCorrT']
            self.spikeMonInpCorrI = npzObject['spikeMonInpCorrI']

    def calculate_spike_rate(self):
        dtHist = float(5 * ms)
        histBins = arange(0, float(self.p['duration']), dtHist)
        histCenters = arange(0 + dtHist / 2, float(self.p['duration']) - dtHist / 2, dtHist)

        FRExc, _ = histogram(self.spikeMonExcT, histBins)
        FRInh, _ = histogram(self.spikeMonInhT, histBins)

        FRExc = FRExc / dtHist / self.p['nExc']
        FRInh = FRInh / dtHist / self.p['nInh']

        self.dtHistFR = dtHist
        self.histBins = histBins
        self.histCenters = histCenters
        self.FRExc = FRExc
        self.FRInh = FRInh

    def calculate_voltage_histogram(self, removeMode=False):
        voltageNumpyExc = self.stateMonExcV[0, :]
        voltageNumpyInh = self.stateMonInhV[0, :]

        if removeMode:
            voltageNumpyExcMode, _ = mode(voltageNumpyExc)
            voltageNumpyInhMode, _ = mode(voltageNumpyInh)
            # keepBoolExc = voltageNumpyExc != voltageNumpyExcMode[0]
            # keepBoolInh = voltageNumpyInh != voltageNumpyInhMode[0]
            keepBoolExc = ~np.isclose(voltageNumpyExc, voltageNumpyExcMode[0])
            keepBoolInh = ~np.isclose(voltageNumpyInh, voltageNumpyInhMode[0])
            voltageNumpyExc = voltageNumpyExc[keepBoolExc]
            voltageNumpyInh = voltageNumpyInh[keepBoolInh]

        allVoltageNumpy = np.concatenate((voltageNumpyExc, voltageNumpyInh))

        voltageHistBins = arange(allVoltageNumpy.min(), allVoltageNumpy.max(), .1)
        voltageHistCenters = bins_to_centers(voltageHistBins)

        voltageHistExc, _ = histogram(voltageNumpyExc, voltageHistBins)
        voltageHistInh, _ = histogram(voltageNumpyInh, voltageHistBins)

        self.voltageHistBins = voltageHistBins
        self.voltageHistCenters = voltageHistCenters
        self.voltageHistExc = voltageHistExc
        self.voltageHistInh = voltageHistInh

    def calculate_upstates(self):
        # we use the firing rate of the inhibtory unit for 2 reasons:
        # using voltage is imprecise because not all units immediately increase voltage
        # the inhibitory unit has a higher FR
        # these settings seem to work well...

        ups, downs = find_upstates(self.FRInh, self.dtHistFR,
                                   v_thresh=.2, dur_thresh=.1, extension_thresh=.1)

        # since we used the FR histogram to calculate the up/down durations, we must convert back by multiplying
        upDurs = (downs - ups) * self.dtHistFR
        downDurs = (ups[1:] - downs[:-1]) * self.dtHistFR
        allDurs = np.concatenate((upDurs, downDurs))

        # make a histogram of state durations
        dtHistUDStateDurs = float(200 * ms)
        if allDurs.size > 0:
            histMaxUDStateDurs = allDurs.max() + dtHistUDStateDurs
        else:
            histMaxUDStateDurs = 5
        histBinsUDStateDurs = arange(0, histMaxUDStateDurs, dtHistUDStateDurs)
        histCentersUDStateDurs = bins_to_centers(histBinsUDStateDurs)

        upDurHist, _ = histogram(upDurs, histBinsUDStateDurs)
        downDurHist, _ = histogram(downDurs, histBinsUDStateDurs)

        self.ups = ups * self.dtHistFR
        self.downs = downs * self.dtHistFR
        self.upDurs = upDurs
        self.downDurs = downDurs
        self.histCentersUDStateDurs = histCentersUDStateDurs
        self.upDurHist = upDurHist
        self.downDurHist = downDurHist

    def reshape_upstates(self):

        voltageNumpyExc = self.stateMonExcV[0, :]
        voltageNumpyInh = self.stateMonInhV[0, :]

        # ups is in units of seconds. to convert to the index in the original voltage recording,
        # we can divide by the DT
        upInds = (self.ups / float(self.p['dt'])).astype(int)

        nUpstates = len(self.ups)

        if nUpstates == 0:
            print('there were no detectable up states')
            return

        # trial limits
        # if len(self.downDurs) > 0:
        #     preTrial = -self.downDurs.min()
        # else:
        #     preTrial = -0.1
        preTrial = -0.1
        postTrial = self.upDurs.max()

        tTrial = np.linspace(preTrial, postTrial, int((postTrial - preTrial) / float(self.p['dt']) + 1))
        tTrial = tTrial[:-1]
        nSampsPerTrial = tTrial.shape[0]
        preTrialSamps = (tTrial < 0).sum()

        # could be zeros or nan, affects how it looks...
        vUpstatesExc = np.full((nUpstates, nSampsPerTrial), np.nan)
        vUpstatesInh = np.full((nUpstates, nSampsPerTrial), np.nan)

        for upstateInd in range(nUpstates):
            startIndRight = int(upInds[upstateInd] - preTrialSamps)
            endIndRight = int(upInds[upstateInd] + nSampsPerTrial - preTrialSamps)

            # check if the indices make sense
            if startIndRight < 0:
                adjustedStartIndRight = 0
                startIndLeft = -startIndRight
            else:
                adjustedStartIndRight = startIndRight
                startIndLeft = 0

            if endIndRight > voltageNumpyExc.shape[0]:
                adjustedEndIndRight = voltageNumpyExc.shape[0]
                endIndLeft = nSampsPerTrial - endIndRight + voltageNumpyExc.shape[0]
            else:
                adjustedEndIndRight = endIndRight
                endIndLeft = nSampsPerTrial

            vUpstatesExc[upstateInd, startIndLeft:endIndLeft] = voltageNumpyExc[
                                                                adjustedStartIndRight:adjustedEndIndRight]
            vUpstatesInh[upstateInd, startIndLeft:endIndLeft] = voltageNumpyInh[
                                                                adjustedStartIndRight:adjustedEndIndRight]

        self.tTrial = tTrial
        self.vUpstatesExc = vUpstatesExc
        self.vUpstatesInh = vUpstatesInh

    def calculate_FR_in_upstates(self):

        ups = self.ups
        downs = self.downs

        nUpstates = len(self.ups)

        if nUpstates == 0:
            print('there were no detectable up states')
            return

        upOnsetRelativeSpikeTimesExc = []
        upOnsetRelativeSpikeTimesInh = []
        upOnsetRelativeSpikeIndicesExc = []
        upOnsetRelativeSpikeIndicesInh = []

        for upstateInd in range(nUpstates):
            inRangeBoolExc = (self.spikeMonExcT > ups[upstateInd]) & (self.spikeMonExcT < downs[upstateInd])
            spikeTimesExc = self.spikeMonExcT[inRangeBoolExc]
            onsetRelativeTimesExc = spikeTimesExc - ups[upstateInd]
            upOnsetRelativeSpikeTimesExc.append(onsetRelativeTimesExc)
            upOnsetRelativeSpikeIndicesExc.append(self.spikeMonExcI[inRangeBoolExc])

            inRangeBoolInh = (self.spikeMonInhT > ups[upstateInd]) & (self.spikeMonInhT < downs[upstateInd])
            spikeTimesInh = self.spikeMonInhT[inRangeBoolInh]
            onsetRelativeTimesInh = spikeTimesInh - ups[upstateInd]
            upOnsetRelativeSpikeTimesInh.append(onsetRelativeTimesInh)
            upOnsetRelativeSpikeIndicesInh.append(self.spikeMonInhI[inRangeBoolInh])

        upOnsetRelativeSpikeTimesExcArray = np.concatenate(upOnsetRelativeSpikeTimesExc)
        upOnsetRelativeSpikeTimesInhArray = np.concatenate(upOnsetRelativeSpikeTimesInh)
        upOnsetRelativeSpikeIndicesExcArray = np.concatenate(upOnsetRelativeSpikeIndicesExc)
        upOnsetRelativeSpikeIndicesInhArray = np.concatenate(upOnsetRelativeSpikeIndicesInh)

        dtHist = float(5 * ms)
        histBins = arange(0, self.upDurs.max(), dtHist)
        histCenters = bins_to_centers(histBins)

        upstateFRExc, _ = histogram(upOnsetRelativeSpikeTimesExcArray, histBins)
        upstateFRInh, _ = histogram(upOnsetRelativeSpikeTimesInhArray, histBins)

        upstateCountsAtBin = zeros(histCenters.shape)
        for binInd, binEdge in enumerate(histCenters):
            upstateCountsAtBin[binInd] = (self.upDurs >= histBins[binInd]).sum()

        upstateFRExc = upstateFRExc / upstateCountsAtBin / self.p['nExc'] / dtHist
        upstateFRInh = upstateFRInh / upstateCountsAtBin / self.p['nInh'] / dtHist

        self.upOnsetRelativeSpikeTimesExcArray = upOnsetRelativeSpikeTimesExcArray
        self.upOnsetRelativeSpikeTimesInhArray = upOnsetRelativeSpikeTimesInhArray
        self.upOnsetRelativeSpikeIndicesExcArray = upOnsetRelativeSpikeIndicesExcArray
        self.upOnsetRelativeSpikeIndicesInhArray = upOnsetRelativeSpikeIndicesInhArray
        self.histCentersUpstateFR = histCenters
        self.upstateFRExc = upstateFRExc
        self.upstateFRInh = upstateFRInh

    def plot_consecutive_state_correlation(self, ax):
        # ax should have 2 elements

        upDurs = self.upDurs
        downDurs = self.downDurs

        OUTLIER_STD_DISTANCE = 2.5

        upDurOutlierBool = np.logical_or(upDurs < upDurs.mean() - OUTLIER_STD_DISTANCE * upDurs.std(),
                                         upDurs > upDurs.mean() + OUTLIER_STD_DISTANCE * upDurs.std())
        downDurOutlierBool = np.logical_or(downDurs < downDurs.mean() - OUTLIER_STD_DISTANCE * downDurs.std(),
                                           downDurs > downDurs.mean() + OUTLIER_STD_DISTANCE * downDurs.std())

        upDursGood = upDurs.copy()
        upDursGood[upDurOutlierBool] = np.nan

        downDursGood = downDurs.copy()
        downDursGood[downDurOutlierBool] = np.nan

        precedingUpDurOutlierBool = upDurOutlierBool[1:]
        subsequentUpDurOutlierBool = upDurOutlierBool[:-1]

        precedingUpDurs = upDurs[1:]
        subsequentUpDurs = upDurs[:-1]
        precedingUpDursGood = upDursGood[1:]
        subsequentUpDursGood = upDursGood[:-1]

        # x = preceding down, y = up
        eitherNan = np.logical_or(np.isnan(downDursGood), np.isnan(precedingUpDursGood))
        print('removing', eitherNan.sum(), 'nan vals')
        pred_x, pred_y, b, r2, p = regress_linear(downDursGood[~eitherNan],
                                                  precedingUpDursGood[~eitherNan])

        combinedOutlierBool = np.logical_or(downDurOutlierBool, precedingUpDurOutlierBool)
        ax[0].scatter(downDurs[~combinedOutlierBool], precedingUpDurs[~combinedOutlierBool])
        ax[0].scatter(downDurs[combinedOutlierBool], precedingUpDurs[combinedOutlierBool], marker='x')
        ax[0].plot(pred_x, pred_y)
        ax[0].set(xlabel='Preceding DownDur', ylabel='UpDur',
                  title='k = 0, r = {:.2f}, p = {:.3f}'.format(np.sign(b) * np.sqrt(r2), p))

        # x = subsequent down, y = up
        eitherNan = np.logical_or(np.isnan(downDursGood), np.isnan(subsequentUpDursGood))
        print('removing', eitherNan.sum(), 'nan vals')
        pred_x, pred_y, b, r2, p = regress_linear(downDursGood[~eitherNan],
                                                  subsequentUpDursGood[~eitherNan])

        combinedOutlierBool = np.logical_or(downDurOutlierBool, subsequentUpDurOutlierBool)
        ax[1].scatter(downDurs[~combinedOutlierBool], subsequentUpDurs[~combinedOutlierBool])
        ax[1].scatter(downDurs[combinedOutlierBool], subsequentUpDurs[combinedOutlierBool], marker='x')
        ax[1].plot(pred_x, pred_y)
        ax[1].set(xlabel='Subsequent DownDur', ylabel='UpDur',
                  title='k = 1, r = {:.2f}, p = {:.3f}'.format(np.sign(b) * np.sqrt(r2), p))

    def plot_spike_raster(self, ax):
        ax.scatter(self.spikeMonExcT, self.spikeMonExcI, c='g', s=1,
                   marker='.', linewidths=0)
        ax.scatter(self.spikeMonInhT, self.p['nExcSpikemon'] + self.spikeMonInhI, c='r', s=1,
                   marker='.', linewidths=0)
        ax.set(xlim=(0., self.p['duration'] / second), ylim=(0, self.p['nUnits']), ylabel='neuron index')

    def plot_firing_rate(self, ax):
        ax.plot(self.histCenters[:self.FRExc.size], self.FRExc, label='Exc', color='green', alpha=0.5)
        ax.plot(self.histCenters[:self.FRInh.size], self.FRInh, label='Inh', color='red', alpha=0.5)
        ax.legend()
        ax.set(xlabel='Time (s)', ylabel='Firing Rate (Hz)')

    def plot_voltage_histogram(self, ax, yScaleLog=False):
        ax.plot(self.voltageHistCenters, self.voltageHistExc, color='green', alpha=0.5)
        ax.plot(self.voltageHistCenters, self.voltageHistInh, color='red', alpha=0.5)
        ax.vlines(self.p['eLeakExc'] / mV, 0, self.voltageHistExc.max() / 2, color='green', ls='--', alpha=0.5)
        ax.vlines(self.p['eLeakInh'] / mV, 0, self.voltageHistInh.max() / 2, color='red', ls='--', alpha=0.5)
        ax.set(xlabel='voltage (mV)', ylabel='# of occurences')

        if yScaleLog:
            plt.yscale('log')

    def plot_voltage_detail(self, ax, unitType='Exc', useStateInd=0, plotKicks=False, **kwargs):
        # reconstruct the time vector
        stateMonT = np.arange(0, float(self.p['duration']), float(self.p['dt']))
        yLims = (self.p['eLeakExc'] / mV - 15, self.p['eLeakExc'] / mV + 70)

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

        useThresh = self.p['vThresh' + unitType] / mV
        # ax.axhline(useThresh, color=useColor, linestyle=':')  # Threshold
        # ax.axhline(self.p['eLeak' + unitType] / mV, color=useColor, linestyle='--')  # Resting
        ax.plot(stateMonT, voltageSeries, color=useColor, lw=.3, **kwargs)

        if plotKicks and 'kickTimes' in self.p:
            kickTimes = array(self.p['kickTimes'])
            ax.scatter(kickTimes, np.ones_like(kickTimes) * self.p['eLeakExc'] / mV - 10)

        if hasattr(self, 'spikeMonInpCorrI'):
            # spikeDict = convert_indices_times_to_dict(self.poissonCorrInputIndices, self.poissonCorrInputTimes)
            yOffset = self.p['eLeakExc'] / mV - 10
            yDist = -20
            maxInd = self.spikeMonInpCorrI.max()
            # for unitInd, spikeTimeArray in spikeDict.items():
            ax.scatter(x=self.spikeMonInpCorrT,
                       y=yOffset + yDist * self.spikeMonInpCorrI / maxInd,
                       c=self.spikeMonInpCorrI,
                       cmap='viridis',
                       s=10, marker='.',  # this makes them quite small!
                       )
            yLims = (self.p['eLeakExc'] / mV - 15 + yDist, self.p['eLeakExc'] / mV + 70)

        if 'indsRecordStateExc' in self.p:
            translatedStateInd = self.p['indsRecordStateExc'][useStateInd]
        else:
            translatedStateInd = useStateInd
        ax.vlines(spikeMonT[spikeMonI == translatedStateInd], useThresh, useThresh + 40, color=useColor, lw=.3,
                  **kwargs)
        ax.set(xlim=(0., self.p['duration'] / second), ylim=yLims, ylabel='mV')

    def plot_updur_lines(self, ax):
        yVal = self.p['eLeakExc'] / mV - 10
        for upTime, downTime in zip(self.ups, self.downs):
            ax.plot([upTime, downTime], [yVal, yVal], c='k')

    def plot_state_duration_histogram(self, ax):
        ax.plot(self.histCentersUDStateDurs[:self.upDurHist.size], self.upDurHist,
                label='Up', color='orange', alpha=0.5)
        ax.plot(self.histCentersUDStateDurs[:self.downDurHist.size], self.downDurHist,
                label='Down', color='purple', alpha=0.5)
        ax.legend()
        ax.set(xlabel='Time (s)', ylabel='# of occurences')

    def plot_upstate_voltages(self, ax):

        yLims = (self.p['eLeakExc'] / mV - 5, self.p['eLeakExc'] / mV + 20)

        # ax.plot(self.tTrial, self.vUpstatesExc.T)
        ax.plot(self.tTrial, np.nanmean(self.vUpstatesExc, 0), color='green')
        ax.plot(self.tTrial, np.nanmean(self.vUpstatesInh, 0), color='red')
        ax.set(xlim=(self.tTrial[0], self.tTrial[-1]), ylim=yLims,
               xlabel='Time (s)', ylabel='Voltage (mV)')

    def plot_upstate_voltage_image(self, ax, sortByDuration=True):

        # sort?
        if sortByDuration:
            durationSortOrder = np.argsort(self.upDurs)
            vUpstatesDurSort = self.vUpstatesExc[durationSortOrder, :]
        else:
            vUpstatesDurSort = self.vUpstatesExc

        i = ax.imshow(vUpstatesDurSort,
                      extent=[self.tTrial[0], self.tTrial[-1], 0, len(self.ups)],
                      aspect='auto',
                      interpolation='none', )
        ax.set(xlabel='Time (s)', ylabel='Upstate Index')
        cb = plt.colorbar(i, ax=ax)
        cb.ax.set_ylabel('Voltage (mV)', rotation=270)

    def plot_upstate_raster(self, ax):
        ax.scatter(self.upOnsetRelativeSpikeTimesExcArray, self.upOnsetRelativeSpikeIndicesExcArray,
                   c='g', s=1, marker='.', linewidths=0)
        ax.scatter(self.upOnsetRelativeSpikeTimesInhArray,
                   self.p['nExcSpikemon'] + self.upOnsetRelativeSpikeIndicesInhArray,
                   c='r', s=1, marker='.', linewidths=0)
        ax.set(xlim=(self.tTrial[0], self.tTrial[-1]), ylim=(0, self.p['nUnits']), ylabel='neuron index')

    def plot_upstate_FR(self, ax):
        ax.plot(self.histCentersUpstateFR[:self.upstateFRExc.size],
                self.upstateFRExc, label='Exc', color='green', alpha=0.5)
        ax.plot(self.histCentersUpstateFR[:self.upstateFRInh.size],
                self.upstateFRInh, label='Inh', color='red', alpha=0.5)
        ax.legend()
        ax.set(xlabel='Time (s)', ylabel='Firing Rate (Hz)')


class ResultsEphys(object):

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
        npzObject = np.load(loadPath, allow_pickle=True)
        self.npzObject = npzObject

        self.spikeMonExcT = npzObject['spikeMonExcT']
        self.spikeMonExcI = npzObject['spikeMonExcI']
        self.spikeMonExcC = npzObject['spikeMonExcC']
        self.spikeMonInhT = npzObject['spikeMonInhT']
        self.spikeMonInhI = npzObject['spikeMonInhI']
        self.spikeMonInhC = npzObject['spikeMonInhC']
        self.stateMonExcV = npzObject['stateMonExcV']
        self.stateMonInhV = npzObject['stateMonInhV']
        self.spikeTrainsExc = npzObject['spikeTrainsExc'][()]
        self.spikeTrainsInh = npzObject['spikeTrainsInh'][()]

    def calculate_and_plot(self, f, ax):
        I_ext_range = self.p['iExtRange']
        ExcData = self.spikeMonExcC / self.p['duration']
        InhData = self.spikeMonInhC / self.p['duration']

        I_index_for_ISI = int(len(I_ext_range) * .9) - 1

        # reconstruct time
        stateMonT = np.arange(0, float(self.p['duration']), float(self.p['dt']))

        # might be useful...
        # ax.axhline(useThresh, color=useColor, linestyle=':')  # Threshold
        # ax.axhline(self.p['eLeak' + unitType] / mV, color=useColor, linestyle='--')  # Resting

        useThresh = self.p['vThreshExc'] / mV
        ax[0, 0].plot(stateMonT, self.stateMonExcV[I_index_for_ISI, :], color='g')
        ax[0, 0].vlines(self.spikeTrainsExc[I_index_for_ISI], useThresh, useThresh + 40, color='g', lw=.3)
        ax[0, 0].set(xlim=(0., self.p['duration'] / second), ylabel='mV', xlabel='Time (s)')

        useThresh = self.p['vThreshInh'] / mV
        ax[0, 1].plot(stateMonT, self.stateMonInhV[I_index_for_ISI, :], color='g')
        ax[0, 1].vlines(self.spikeTrainsInh[I_index_for_ISI], useThresh, useThresh + 40, color='g', lw=.3)
        ax[0, 1].set(xlim=(0., self.p['duration'] / second), ylabel='mV', xlabel='Time (s)')

        ax[1, 0].plot(I_ext_range * 1e9, ExcData, label='Exc')
        ax[1, 0].plot(I_ext_range * 1e9, InhData, label='Inh')
        ax[1, 0].axvline(float(I_ext_range[I_index_for_ISI]) * 1e9,
                         label='displayed value', color='grey', ls='--')
        ax[1, 0].set_xlabel('Current (nA)')
        ax[1, 0].set_ylabel('Firing Rate (Hz)')
        ax[1, 0].legend()

        ISIExc = diff(self.spikeTrainsExc[I_index_for_ISI])
        ISIInh = diff(self.spikeTrainsInh[I_index_for_ISI])
        ax[1, 1].plot(arange(1, len(ISIExc) + 1), ISIExc * 1000, label='Exc')
        ax[1, 1].plot(arange(1, len(ISIInh) + 1), ISIInh * 1000, label='Inh')
        ax[1, 1].set_xlabel('ISI number')
        ax[1, 1].set_ylabel('ISI (ms)')
        ax[1, 1].legend()

        f.tight_layout()
        f.subplots_adjust(top=.9)
