"""
classes for representing the results of simulations.
i.e. spikes and unit state variables (membrane voltage, synaptic variables, etc)
"""

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

    def __init__(self):
        pass

    def init_from_file(self, resultsIdentifier, loadFolder):

        self.loadFolder = loadFolder
        self.interpret_rID(resultsIdentifier)
        self.load_params_from_file()
        self.load_results_from_file()
        self.timeArray = np.arange(0, float(self.p['duration']), float(self.p['dt']))

    def interpret_rID(self, resultsIdentifier):

        fileParts = resultsIdentifier.split('.')

        if len(fileParts) == 1:  # indicates only results id, not a file
            fileNameParts = fileParts[0].split('_')

            if fileNameParts[-1] == 'results':
                self.rID = '_'.join(fileNameParts[:-1])
                self.resultsFileName = resultsIdentifier + '.npz'
                self.paramsFileName = self.rID + '_params.pkl'
            elif fileNameParts[-1] == 'params':
                self.rID = '_'.join(fileNameParts[:-1])
                self.resultsFileName = self.rID + '_results.npz'
                self.paramsFileName = resultsIdentifier + '.pkl'
            else:
                self.rID = resultsIdentifier
                self.resultsFileName = resultsIdentifier + '_results.npz'
                self.paramsFileName = resultsIdentifier + '_params.pkl'

        elif len(fileParts) == 2:  # indicates a file, either pkl or npz

            fileNameParts = fileParts[0].split('_')

            self.rID = fileNameParts[0] + fileNameParts[1]
            if fileParts[1] == 'npz':
                self.resultsFileName = resultsIdentifier
                self.paramsFileName = self.rID + '_params.pkl'
            elif fileParts[1] == 'pkl':
                self.resultsFileName = self.rID + '_results.npz'
                self.paramsFileName = resultsIdentifier

    def load_params_from_file(self):
        loadPath = os.path.join(self.loadFolder, self.paramsFileName)
        with open(loadPath, 'rb') as f:
            params = dill.load(f)
        self.p = params

    def load_results_from_file(self):
        loadPath = os.path.join(self.loadFolder, self.resultsFileName)
        npzObject = np.load(loadPath, allow_pickle=True)
        self.npzObject = npzObject

        # simply assign each object name to an attribute of the results object
        for savedObjectName in npzObject.files:
            setattr(self, savedObjectName, npzObject[savedObjectName])

    def init_from_network_object(self, network_object):
        self.rID = network_object.saveName
        self.p = network_object.p

        # load results from network object
        useDType = np.single

        self.spikeMonExcT = np.array(network_object.spikeMonExc.t, dtype=useDType)
        self.spikeMonExcI = np.array(network_object.spikeMonExc.i, dtype=useDType)
        self.spikeMonInhT = np.array(network_object.spikeMonInh.t, dtype=useDType)
        self.spikeMonInhI = np.array(network_object.spikeMonInh.i, dtype=useDType)
        self.stateMonExcV = np.array(network_object.stateMonExc.v / mV, dtype=useDType)
        self.stateMonInhV = np.array(network_object.stateMonInh.v / mV, dtype=useDType)

        self.timeArray = np.arange(0, float(self.p['duration']), float(self.p['dt']))

    def calculate_PSTH(self):
        dtHist = float(self.p['dtHistPSTH'])
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

    def calculate_avgFR_units(self):

        spikesPerUnitExc = np.bincount(self.spikeMonExcI.astype(int), minlength=self.p['nExc'])
        spikesPerUnitInh = np.bincount(self.spikeMonInhI.astype(int), minlength=self.p['nInh'])

        self.avgFRExcUnits = spikesPerUnitExc / self.p['duration'] / second
        self.avgFRInhUnits = spikesPerUnitInh / self.p['duration'] / second

    def calculate_voltage_histogram(self, removeMode=False, removeReset=False, useAllRecordedUnits=False, useExcUnits=0, useInhUnits=0):
        if useAllRecordedUnits:
            voltageNumpyExc = self.stateMonExcV[:].ravel()
            voltageNumpyInh = self.stateMonInhV[:].ravel()
        else:
            voltageNumpyExc = self.stateMonExcV[useExcUnits, :]
            voltageNumpyInh = self.stateMonInhV[useInhUnits, :]

        if removeMode:
            voltageNumpyExcMode, _ = mode(voltageNumpyExc)
            voltageNumpyInhMode, _ = mode(voltageNumpyInh)
            # keepBoolExc = voltageNumpyExc != voltageNumpyExcMode[0]
            # keepBoolInh = voltageNumpyInh != voltageNumpyInhMode[0]
            keepBoolExc = ~np.isclose(voltageNumpyExc, voltageNumpyExcMode[0])
            keepBoolInh = ~np.isclose(voltageNumpyInh, voltageNumpyInhMode[0])
            voltageNumpyExc = voltageNumpyExc[keepBoolExc]
            voltageNumpyInh = voltageNumpyInh[keepBoolInh]

        if removeReset:
            voltageNumpyExcReset = self.p['vResetExc'] / mV
            voltageNumpyInhReset = self.p['vResetInh'] / mV
            # keepBoolExc = voltageNumpyExc != voltageNumpyExcReset[0]
            # keepBoolInh = voltageNumpyInh != voltageNumpyInhReset[0]
            keepBoolExc = ~np.isclose(voltageNumpyExc, voltageNumpyExcReset, rtol=1e-04, atol=1e-07)
            keepBoolInh = ~np.isclose(voltageNumpyInh, voltageNumpyInhReset, rtol=1e-04, atol=1e-07)
            voltageNumpyExc = voltageNumpyExc[keepBoolExc]
            voltageNumpyInh = voltageNumpyInh[keepBoolInh]

        allVoltageNumpy = np.concatenate((voltageNumpyExc, voltageNumpyInh))

        # print(allVoltageNumpy.min(), allVoltageNumpy.max())
        useMax = np.max((self.p['vThreshExc'] / mV, self.p['vThreshInh'] / mV))
        voltageHistBins = np.arange(allVoltageNumpy.min(), useMax, .1)
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
                                   v_thresh=.2, dur_thresh=.1, extension_thresh=.1,
                                   last_up_must_end=False)

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
        self.upstateFRExcHist = upstateFRExc
        self.upstateFRInhHist = upstateFRInh

    def calculate_FR_in_upstates_simply(self):

        # here we calculate by simply counting the spikes in the Up state
        # and dividing by the duration

        ups = self.ups
        downs = self.downs

        nUpstates = len(self.ups)

        if nUpstates == 0:
            print('there were no detectable up states')
            return

        upstateFRExc = []
        upstateFRInh = []

        for upstateInd in range(nUpstates):
            inRangeBoolExc = (self.spikeMonExcT > ups[upstateInd]) & (
                        self.spikeMonExcT < downs[upstateInd])
            upstateFRExc.append(inRangeBoolExc.sum() / self.p['nExc'] / self.upDurs[upstateInd])

            inRangeBoolInh = (self.spikeMonInhT > ups[upstateInd]) & (
                        self.spikeMonInhT < downs[upstateInd])
            upstateFRInh.append(inRangeBoolInh.sum() / self.p['nInh'] / self.upDurs[upstateInd])

        self.upstateFRExc = np.array(upstateFRExc)
        self.upstateFRInh = np.array(upstateFRInh)

    def calculate_upFR_units(self):

        # here we calculate by simply counting the spikes in the Up state
        # and dividing by the duration

        ups = self.ups
        downs = self.downs

        nUpstates = len(self.ups)

        if nUpstates == 0:
            print('there were no detectable up states')
            return

        upstateFRExc = []
        upstateFRInh = []
        upstateFRExcUnits = np.empty((nUpstates, self.p['nExc']))
        upstateFRInhUnits = np.empty((nUpstates, self.p['nInh']))

        for upstateInd in range(nUpstates):
            inRangeBoolExc = (self.spikeMonExcT > ups[upstateInd]) & (
                        self.spikeMonExcT < downs[upstateInd])
            upstateFRExc.append(inRangeBoolExc.sum() / self.p['nExc'] / self.upDurs[upstateInd])
            indicesInRangeExc = self.spikeMonExcI[inRangeBoolExc].astype(int)
            spikesPerUnitExc = np.bincount(indicesInRangeExc, minlength=self.p['nExc'])
            upstateFRExcUnits[upstateInd, :] = spikesPerUnitExc / self.upDurs[upstateInd]

            inRangeBoolInh = (self.spikeMonInhT > ups[upstateInd]) & (
                        self.spikeMonInhT < downs[upstateInd])
            upstateFRInh.append(inRangeBoolInh.sum() / self.p['nInh'] / self.upDurs[upstateInd])
            indicesInRangeInh = self.spikeMonInhI[inRangeBoolInh].astype(int)
            spikesPerUnitInh = np.bincount(indicesInRangeInh, minlength=self.p['nInh'])
            upstateFRInhUnits[upstateInd, :] = spikesPerUnitInh / self.upDurs[upstateInd]

        self.upstateFRExc = np.array(upstateFRExc)
        self.upstateFRInh = np.array(upstateFRInh)
        self.upstateFRExcUnits = upstateFRExcUnits
        self.upstateFRInhUnits = upstateFRInhUnits

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

    def plot_spike_raster(self, ax, downSampleUnits=True, rng=None):

        if not rng:
            rng = np.random.default_rng(None)  # random seed

        if downSampleUnits:
            targetDisplayedExcUnits = 160
            targetDisplayedInhUnits = 40
            downSampleE = rng.choice(self.p['nExc'], size=targetDisplayedExcUnits, replace=False)
            downSampleI = rng.choice(self.p['nInh'], size=targetDisplayedInhUnits, replace=False)
            matchingEUnitsBool = np.isin(self.spikeMonExcI, downSampleE)
            matchingIUnitsBool = np.isin(self.spikeMonInhI, downSampleI)
            DownSampleERev = np.full((downSampleE.max() + 1,), np.nan)
            DownSampleERev[downSampleE] = np.arange(downSampleE.size)
            DownSampleIRev = np.full((downSampleI.max() + 1,), np.nan)
            DownSampleIRev[downSampleI] = np.arange(downSampleI.size)
            xExc = self.spikeMonExcT[matchingEUnitsBool]
            yExc = DownSampleERev[self.spikeMonExcI[matchingEUnitsBool].astype(int)]
            xInh = self.spikeMonInhT[matchingIUnitsBool]
            yInh = targetDisplayedExcUnits + DownSampleIRev[self.spikeMonInhI[matchingIUnitsBool].astype(int)]
            yLims = (0, targetDisplayedExcUnits + targetDisplayedInhUnits)
        else:
            xExc = self.spikeMonExcT
            yExc = self.spikeMonExcI
            xInh = self.spikeMonInhT
            yInh = self.p['nExc'] + self.spikeMonInhI
            yLims = (0, self.p['nUnits'])

        ax.scatter(xExc, yExc, c='cyan', s=1, marker='.', linewidths=0)
        ax.scatter(xInh, yInh, c='red', s=1, marker='.', linewidths=0)
        ax.set(xlim=(0., self.p['duration'] / second), ylim=yLims, ylabel='neuron index')  # ylim=(0, self.p['nUnits']),

        if 'upPoissonTimes' in self.p:
            ax.vlines(self.p['upPoissonTimes'], *ax.get_ylim(), color='k', linestyles='--', alpha=0.4)

    def plot_firing_rate(self, ax):
        ax.plot(self.histCenters[:self.FRExc.size], self.FRExc, label='Exc', color='cyan', alpha=0.5)
        ax.plot(self.histCenters[:self.FRInh.size], self.FRInh, label='Inh', color='red', alpha=0.5)
        # ax.legend()
        ax.set(ylabel='Firing Rate (Hz)')

    def plot_voltage_histogram(self, ax, yScaleLog=False):
        ax.plot(self.voltageHistCenters, self.voltageHistExc, color='green', alpha=0.5)
        ax.plot(self.voltageHistCenters, self.voltageHistInh, color='red', alpha=0.5)
        ax.vlines(self.p['eLeakExc'] / mV, 0, self.voltageHistExc.max() / 2, color='green', ls='--', alpha=0.5)
        ax.vlines(self.p['eLeakInh'] / mV, 0, self.voltageHistInh.max() / 2, color='red', ls='--', alpha=0.5)
        ax.set(xlabel='voltage (mV)', ylabel='# of occurences')

        if yScaleLog:
            plt.yscale('log')

    def plot_voltage_histogram_sideways(self, ax, unitType='Exc', yScaleLog=False, moveHalfwayUp=True):

        # find out what the x axis is... the far right-hand 10% will act as our "y axis"
        xMin, xMax = ax.get_xlim()
        yMin = xMin + (xMax - xMin) * .9
        yMax = xMax

        if moveHalfwayUp:
            xVals = self.voltageHistCenters + 40
        else:
            xVals = self.voltageHistCenters

        if unitType == 'Exc':
            yVals = self.voltageHistExc
            useColor = 'green'
            useELeak = self.p['eLeakExc'] / mV
        elif unitType == 'Inh':
            yVals = self.voltageHistInh
            useColor = 'red'
            useELeak = self.p['eLeakInh'] / mV
        else:
            print('the unit type request doesnt exist')
            return

        # rescale the yVals to work with the min and max we found
        yValsRescaled = yMin + yVals / yVals.max() * (yMax - yMin)

        # if you want the y to be log-scaled, take the log base 10 of the y values
        ax.plot(yValsRescaled, xVals, color=useColor, alpha=0.5)
        # ax.hlines(useELeak, 0, yVals.max() / 2, color=useColor, ls='--', alpha=0.5)

    def plot_voltage_detail(self, ax, unitType='Exc', useStateInd=0, plotKicks=False, overrideColor='', **kwargs):

        useLineWidth = 0.5
        useSpikeAmp = 0
        downSampleBy = 10

        if 'stateVariableDT' in self.p and self.p['recordAllVoltage']:
            useDT = self.p['stateVariableDT']
        else:
            useDT = self.p['dt']

        # reconstruct the time vector
        stateMonT = np.arange(0, float(self.p['duration']), float(useDT * downSampleBy))
        yLims = (self.p['eLeakExc'] / mV - 15, self.p['eLeakExc'] / mV + 70)

        if unitType == 'Exc':
            voltageSeries = self.stateMonExcV[useStateInd, ::downSampleBy]
            spikeMonT = self.spikeMonExcT
            spikeMonI = self.spikeMonExcI
            useColor = 'cyan'
            if 'indsRecordStateExc' in self.p:
                translatedStateInd = self.p['indsRecordStateExc'][useStateInd]
            else:
                translatedStateInd = useStateInd
        elif unitType == 'Inh':
            voltageSeries = self.stateMonInhV[useStateInd, ::downSampleBy]
            spikeMonT = self.spikeMonInhT
            spikeMonI = self.spikeMonInhI
            useColor = 'red'
            if 'indsRecordStateInh' in self.p:
                translatedStateInd = self.p['indsRecordStateInh'][useStateInd]
            else:
                translatedStateInd = useStateInd
        else:
            print('you messed up')
            return

        if overrideColor:
            useColor = overrideColor

        if len(stateMonT) > len(voltageSeries):
            stateMonT = stateMonT[1:]

        useThresh = self.p['vThresh' + unitType] / mV
        # ax.axhline(useThresh, color=useColor, linestyle=':')  # Threshold
        # ax.axhline(self.p['eLeak' + unitType] / mV, color=useColor, linestyle='--')  # Resting
        ax.plot(stateMonT, voltageSeries, color=useColor, lw=useLineWidth, **kwargs)

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

        ax.vlines(spikeMonT[spikeMonI == translatedStateInd], useThresh, useSpikeAmp, color=useColor, lw=useLineWidth,
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

    def __init__(self):
        pass

    def init_from_file(self, resultsIdentifier, loadFolder):
        self.rID = resultsIdentifier
        self.loadFolder = loadFolder
        self.load_params_from_file()
        self.load_results_from_file()

    def load_params_from_file(self):
        loadPath = os.path.join(self.loadFolder, self.rID + '_params.pkl')
        with open(loadPath, 'rb') as f:
            params = dill.load(f)
        self.p = params

    def load_results_from_file(self):
        loadPath = os.path.join(self.loadFolder, self.rID + '_results.npz')
        npzObject = np.load(loadPath, allow_pickle=True)
        self.npzObject = npzObject

        self.spikeMonExcT = npzObject['spikeMonExcT']
        self.spikeMonExcI = npzObject['spikeMonExcI']
        self.spikeMonExcC = npzObject['spikeMonExcC']
        self.stateMonExcV = npzObject['stateMonExcV']
        self.spikeTrainsExc = npzObject['spikeTrainsExc'][()]

        self.spikeMonInhT = npzObject['spikeMonInhT']
        self.spikeMonInhI = npzObject['spikeMonInhI']
        self.spikeMonInhC = npzObject['spikeMonInhC']
        self.stateMonInhV = npzObject['stateMonInhV']
        self.spikeTrainsInh = npzObject['spikeTrainsInh'][()]

        if 'useSecondPopExc' in self.p and self.p['useSecondPopExc']:
            self.spikeMonExc2T = npzObject['spikeMonExc2T']
            self.spikeMonExc2I = npzObject['spikeMonExc2I']
            self.spikeMonExc2C = npzObject['spikeMonExc2C']
            self.stateMonExc2V = npzObject['stateMonExc2V']
            self.spikeTrainsExc2 = npzObject['spikeTrainsExc2'][()]

    def init_from_network_object(self, N):
        self.rID = N.saveName
        self.p = N.p

        # load results from network object
        useDType = np.single

        self.spikeMonExcT = np.array(N.spikeMonExc.t, dtype=useDType)
        self.spikeMonExcI = np.array(N.spikeMonExc.i, dtype=useDType)
        self.spikeMonExcC = np.array(N.spikeMonExc.count, dtype=useDType)
        self.stateMonExcV = np.array(N.stateMonExc.v / mV, dtype=useDType)
        self.spikeTrainsExc = np.array(N.spikeMonExc.spike_trains(), dtype=object)

        self.spikeMonInhT = np.array(N.spikeMonInh.t, dtype=useDType)
        self.spikeMonInhI = np.array(N.spikeMonInh.i, dtype=useDType)
        self.spikeMonInhC = np.array(N.spikeMonInh.count, dtype=useDType)
        self.stateMonInhV = np.array(N.stateMonInh.v / mV, dtype=useDType)
        self.spikeTrainsInh = np.array(N.spikeMonInh.spike_trains(), dtype=object)

        # if self.p['useSecondPopExc']:
        #     self.spikeMonExc2T = np.array(N.spikeMonExc2.t, dtype=useDType)
        #     self.spikeMonExc2I = np.array(N.spikeMonExc2.i, dtype=useDType)
        #     self.spikeMonExc2C = np.array(N.spikeMonExc2.count, dtype=useDType)
        #     self.stateMonExc2V = np.array(N.stateMonExc2.v / mV, dtype=useDType)
        #     self.spikeTrainsExc2 = np.array(N.spikeMonExc2.spike_trains(), dtype=object)

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
        # ax[0, 0].vlines(self.spikeTrainsExc[()][I_index_for_ISI], useThresh, useThresh + 40, color='g', lw=.3)
        # ax[0, 0].vlines(self.spikeTrainsExc[I_index_for_ISI], useThresh, useThresh + 40, color='g', lw=.3)
        ax[0, 0].vlines(self.spikeTrainsExc[I_index_for_ISI] / second, useThresh, useThresh + 40, color='g', lw=.3)
        ax[0, 0].set(xlim=(0., self.p['duration'] / second), ylabel='mV', xlabel='Time (s)')

        useThresh = self.p['vThreshInh'] / mV
        ax[0, 1].plot(stateMonT, self.stateMonInhV[I_index_for_ISI, :], color='r')
        # ax[0, 1].vlines(self.spikeTrainsInh[()][I_index_for_ISI], useThresh, useThresh + 40, color='g', lw=.3)
        # ax[0, 1].vlines(self.spikeTrainsInh[I_index_for_ISI], useThresh, useThresh + 40, color='r', lw=.3)
        ax[0, 1].vlines(self.spikeTrainsInh[I_index_for_ISI] / second, useThresh, useThresh + 40, color='r', lw=.3)
        ax[0, 1].set(xlim=(0., self.p['duration'] / second), ylabel='mV', xlabel='Time (s)')

        ax[1, 0].plot(I_ext_range * 1e9, ExcData, label='Exc')
        ax[1, 0].plot(I_ext_range * 1e9, InhData, label='Inh')
        ax[1, 0].axvline(float(I_ext_range[I_index_for_ISI]) * 1e9,
                         label='displayed value', color='grey', ls='--')
        ax[1, 0].set_xlabel('Current (nA)')
        ax[1, 0].set_ylabel('Firing Rate (Hz)')
        # ax[1, 0].legend()

        # ISIExc = diff(self.spikeTrainsExc[()][I_index_for_ISI])
        # ISIInh = diff(self.spikeTrainsInh[()][I_index_for_ISI])
        ISIExc = diff(self.spikeTrainsExc[I_index_for_ISI])
        ISIInh = diff(self.spikeTrainsInh[I_index_for_ISI])
        ax[1, 1].plot(arange(1, len(ISIExc) + 1), ISIExc * 1000, label='Exc')
        ax[1, 1].plot(arange(1, len(ISIInh) + 1), ISIInh * 1000, label='Inh')
        ax[1, 1].set_xlabel('ISI number')
        ax[1, 1].set_ylabel('ISI (ms)')
        ax[1, 1].legend()

        f.tight_layout()
        f.subplots_adjust(top=.9)

    def calculate_and_plot_secondExcPop(self, f, ax):
        I_ext_range = self.p['iExtRange']

        ExcData = self.spikeMonExcC / self.p['duration']
        Exc2Data = self.spikeMonExc2C / self.p['duration']
        InhData = self.spikeMonInhC / self.p['duration']

        I_index_for_ISI = int(len(I_ext_range) * .9) - 1
        # plotVoltageForCurrentValues = (0.11, 0.16, 0.21)
        plotVoltageForCurrentValues = (-.1, 0.11, 0.16, 0.21)
        iIndicesToPlot = []
        for pVFCV in plotVoltageForCurrentValues:
            iIndicesToPlot.append(np.argmin(np.fabs(I_ext_range - pVFCV * nA)))

        # reconstruct time
        stateMonT = np.arange(0, float(self.p['duration']), float(self.p['dt']))

        # might be useful...
        # ax.axhline(useThresh, color=useColor, linestyle=':')  # Threshold
        # ax.axhline(self.p['eLeak' + unitType] / mV, color=useColor, linestyle='--')  # Resting

        useYMin = -78
        useYMax = 2
        useSpikeAmp = -20
        useLineWidth = 1

        excColor = 'cyan'
        exc2Color = 'blue'
        inhColor = 'red'

        useThreshExc = self.p['vThreshExc'] / mV
        useThreshExc2 = self.p['vThreshExc2'] / mV
        useThreshInh = self.p['vThreshInh'] / mV

        # excColors = ('lightgreen', 'green', 'darkgreen')
        # exc2Colors = ('paleturquoise', 'turquoise', 'darkturquoise')
        # inhColors = ('tomato', 'red', 'darkred')

        # excColors = ('cyan', 'darkorange', 'darkgoldenrod', 'mediumaquamarine')
        # exc2Colors = ('blue', 'olive', 'lightseagreen', 'royalblue')
        # inhColors = ('red', 'mediumslateblue', 'purple', 'crimson')

        # excColors = ('#00FFFF', '#FF4000', '#FFBF00', '#8000FF')
        # exc2Colors = ('#0000FF', '#FF8000', '#FFFF00', '#FF00FF')
        # inhColors = ('#FF0000', '#00FF00', '#0000FF', '#FF8000')

        # excColors = ('#00FFFF', '#0000FF', '#00FF00')
        # exc2Colors = ('#0000FF', '#8000FF', '#00FFFF')
        # inhColors = ('#FF0000', '#FF4000', '#FF0080')

        for iDummy, iIndexToPlot in enumerate(iIndicesToPlot):
            # excSubColor = excColors[iDummy]
            # exc2SubColor = exc2Colors[iDummy]
            # inhSubColor = inhColors[iDummy]

            excSubColor = excColor
            exc2SubColor = exc2Color
            inhSubColor = inhColor

            ax[0, 0].plot(stateMonT, self.stateMonExcV[iIndexToPlot, :], color=excSubColor, lw=useLineWidth)
            ax[0, 0].vlines(self.spikeTrainsExc[iIndexToPlot] / second, useThreshExc, useSpikeAmp, color=excSubColor,
                            lw=useLineWidth)
            ax[0, 0].set(xlim=(0., self.p['duration'] / second), ylim=(useYMin, useYMax), ylabel='mV', xlabel='Time (s)')

            ax[0, 1].plot(stateMonT, self.stateMonExc2V[iIndexToPlot, :], color=exc2SubColor, lw=useLineWidth)
            ax[0, 1].vlines(self.spikeTrainsExc2[iIndexToPlot] / second, useThreshExc2, useSpikeAmp, color=exc2SubColor,
                            lw=useLineWidth)
            ax[0, 1].set(xlim=(0., self.p['duration'] / second), ylim=(useYMin, useYMax), ylabel='mV', xlabel='Time (s)')

            ax[0, 2].plot(stateMonT, self.stateMonInhV[iIndexToPlot, :], color=inhSubColor, lw=useLineWidth)
            ax[0, 2].vlines(self.spikeTrainsInh[iIndexToPlot] / second, useThreshInh, useSpikeAmp, color=inhSubColor,
                            lw=useLineWidth)
            ax[0, 2].set(xlim=(0., self.p['duration'] / second), ylim=(useYMin, useYMax), ylabel='mV', xlabel='Time (s)')

        ax[1, 0].plot(I_ext_range * 1e9, ExcData, label='Exc', color=excColor)
        ax[1, 0].plot(I_ext_range * 1e9, Exc2Data, label='Exc22', color=exc2Color)
        ax[1, 0].plot(I_ext_range * 1e9, InhData, label='Inh', color=inhColor)
        # ax[1, 0].axvline(float(I_ext_range[I_index_for_ISI]) * 1e9,
        #                  label='displayed value', color='grey', ls='--')
        ax[1, 0].set_xlabel('Current (nA)')
        ax[1, 0].set_ylabel('Firing Rate (Hz)')
        # ax[1, 0].legend()

        # ISIExc = diff(self.spikeTrainsExc[()][I_index_for_ISI])
        # ISIInh = diff(self.spikeTrainsInh[()][I_index_for_ISI])
        ISIExc = diff(self.spikeTrainsExc[I_index_for_ISI])
        ISIExc2 = diff(self.spikeTrainsExc2[I_index_for_ISI])
        ISIInh = diff(self.spikeTrainsInh[I_index_for_ISI])
        ax[1, 1].plot(arange(1, len(ISIExc) + 1), ISIExc * 1000, label='Exc', color=excColor)
        ax[1, 1].plot(arange(1, len(ISIExc2) + 1), ISIExc2 * 1000, label='Exc2', color=exc2Color)
        ax[1, 1].plot(arange(1, len(ISIInh) + 1), ISIInh * 1000, label='Inh', color=inhColor)
        ax[1, 1].set_xlabel('ISI number')
        ax[1, 1].set_ylabel('ISI (ms)')
        ax[1, 1].legend()

        f.tight_layout()
        f.subplots_adjust(top=.9)

    def calculate_thresh_and_gain(self):
        I_ext_range = self.p['iExtRange']

        ExcData = self.spikeMonExcC / self.p['duration']
        firstSpikeIndExc = np.where(ExcData)[0][0]
        threshExc = I_ext_range[firstSpikeIndExc]
        gainRiseExc = ExcData[-1] - ExcData[firstSpikeIndExc - 1]
        gainRunExc = I_ext_range[-1] - I_ext_range[firstSpikeIndExc - 1]
        gainExc = gainRiseExc / gainRunExc

        InhData = self.spikeMonInhC / self.p['duration']
        firstSpikeIndInh = np.where(InhData)[0][0]
        threshInh = I_ext_range[firstSpikeIndInh]
        gainRiseInh = InhData[-1] - InhData[firstSpikeIndInh - 1]
        gainRunInh = I_ext_range[-1] - I_ext_range[firstSpikeIndInh - 1]
        gainInh = gainRiseInh / gainRunInh

        self.threshExc = threshExc
        self.gainExc = gainExc
        self.threshInh = threshInh
        self.gainInh = gainInh

        if self.p['useSecondPopExc']:
            Exc2Data = self.spikeMonExc2C / self.p['duration']
            firstSpikeIndExc2 = np.where(Exc2Data)[0][0]
            threshExc2 = I_ext_range[firstSpikeIndExc2]
            gainRiseExc2 = Exc2Data[-1] - Exc2Data[firstSpikeIndExc2 - 1]
            gainRunExc2 = I_ext_range[-1] - I_ext_range[firstSpikeIndExc2 - 1]
            gainExc2 = gainRiseExc2 / gainRunExc2
            self.threshExc2 = threshExc2
            self.gainExc2 = gainExc2
