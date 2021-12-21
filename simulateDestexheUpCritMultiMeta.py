from brian2 import defaultclock, ms, pA, nA, Hz, seed, mV, second, uS
from params import paramsDestexhe as p
from params import paramsDestexheEphysBuono, paramsDestexheEphysOrig
import numpy as np
from generate import convert_kicks_to_current_series, norm_weights, weight_matrix_from_flat_inds_weights, square_bumps
from trainer import DestexheTrainer
from results import Results

import pandas as pd
import os

rngSeed = None
defaultclock.dt = p['dt']

p['useNewEphysParams'] = False
p['useSecondPopExc'] = False
p['manipulateConnectivity'] = False
p['removePropConn'] = 0.2
p['addBackRemovedConns'] = True
p['paradoxicalKickInh'] = True
weightMult = 1

# PARADOXICAL EFFECT EXPERIMENT PARAMETERS
paradoxicalKickProp = 1
paradoxicalKickTimes = [3000 * ms]
paradoxicalKickDurs = [1000 * ms]
paradoxicalKickSizes = [1]
# paradoxicalKickAmps = np.arange(0, 4.5, 0.5) * pA
# paradoxicalKickAmps = np.arange(9, 17) * pA
paradoxicalKickAmps = np.arange(0, 601, 50) * pA
# paradoxicalKickAmps = np.arange(0, 21, 5) * pA
# paradoxicalKickAmps = np.array([0, 8]) * pA

comparisonStartTime = 2000 * ms
comparisonEndTime = 3000 * ms

p['propConnectFeedforwardProjectionUncorr'] = 0.05  # proportion of feedforward projections that are connected
p['nPoissonUncorrInputUnits'] = 2000
p['nUncorrFeedforwardSynapsesPerUnit'] = int(p['propConnectFeedforwardProjectionUncorr'] * 2000 * (1 - p['propInh']))
p['poissonUncorrInputRate'] = 2 * p['nUncorrFeedforwardSynapsesPerUnit'] * Hz
p['qExcFeedforwardUncorr'] = 0.6 * uS / p['nUncorrFeedforwardSynapsesPerUnit']
p['duration'] = 5.1 * second

p['useRule'] = 'upCrit'
p['nameSuffix'] = 'multiMetaDestexheAI3'
p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True
p['useOldWeightMagnitude'] = True
p['disableWeightScaling'] = True
p['applyLogToFR'] = False
p['setMinimumBasedOnBalance'] = False
p['recordMovieVariables'] = False
p['downSampleVoltageTo'] = 1 * ms
p['stateVariableDT'] = 1 * ms
p['dtHistPSTH'] = 10 * ms
p['recordAllVoltage'] = True

# simulation params
p['nUnits'] = 2e3
p['propConnect'] = 0.25

# p['initWeightMethod'] = 'guessBuono7Weights2e3p025'
# p['initWeightMethod'] = 'guessBuono7Weights2e3p025SlightLowTuned'
p['initWeightMethod'] = 'default'
p['initWeightPrior'] = ''
p['kickType'] = 'barrage'  # kick or spike
p['nUnitsToSpike'] = int(np.round(0.05 * p['nUnits']))
p['timeToSpike'] = 100 * ms
p['timeAfterSpiked'] = paradoxicalKickTimes[-1] + paradoxicalKickDurs[-1] + 1000 * ms
p['spikeInputAmplitude'] = 0.96  # in nA
p['allowAutapses'] = False

p['nUnitsSecondPopExc'] = int(np.round(0.05 * p['nUnits']))
p['startIndSecondPopExc'] = p['nUnitsToSpike']

# boring params
p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])
indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)
p['nExc'] = int(p['nUnits'] * (1 - p['propInh']))
p['nInh'] = int(p['nUnits'] * p['propInh'])

if p['recordAllVoltage']:
    p['indsRecordStateExc'] = list(range(p['nExc']))
    p['indsRecordStateInh'] = list(range(p['nInh']))

if not p['paradoxicalKickInh']:
    p['propKicked'] = 0.1
    p['onlyKickExc'] = True
    p['kickTimes'] = [100 * ms]
    p['kickSizes'] = [1]
    iKickRecorded = convert_kicks_to_current_series(p['kickDur'], p['kickTau'],
                                                    p['kickTimes'], p['kickSizes'], p['duration'], p['dt'])
    p['iKickRecorded'] = iKickRecorded

# END OF PARAMS
nTrials = 30
nCurrentValues = paradoxicalKickAmps.size

paraFRAvgExc = np.empty((nTrials,))
paraFRAvgInh = np.empty((nTrials,))
compFRAvgExc = np.empty((nTrials,))
compFRAvgInh = np.empty((nTrials,))

estimatedDuration = paradoxicalKickTimes[-1] + paradoxicalKickDurs[-1] + p['timeToSpike'] + 1000 * ms
estimatedNBinCenters = int(np.round(estimatedDuration / p['dtHistPSTH'])) - 1

PSTHExc = np.empty((nCurrentValues, nTrials, estimatedNBinCenters))
PSTHInh = np.empty((nCurrentValues, nTrials, estimatedNBinCenters))

overallDF = pd.DataFrame()

for currentValueInd, currentValue in enumerate(paradoxicalKickAmps):
    paradoxicalKickAmp = currentValue

    for trialInd in range(nTrials):

        pL = p.copy()  # pL stands for p in the loop

        pL['nameSuffix'] = pL['nameSuffix'] + str(trialInd)

        # set RNG seeds...
        rngSeed = trialInd
        pL['rngSeed'] = rngSeed
        rng = np.random.default_rng(rngSeed)  # for numpy
        seed(rngSeed)  # for Brian... will insert code to set the random number generator seed into the generated code
        pL['rng'] = rng

        DT = DestexheTrainer(pL)
        DT.set_up_network_upCrit(priorResults=None, recordAllVoltage=pL['recordAllVoltage'])

        if pL['paradoxicalKickInh']:
            DT.p['propKicked'] = paradoxicalKickProp
            DT.p['kickTimes'] = paradoxicalKickTimes
            DT.p['kickDurs'] = paradoxicalKickDurs
            DT.p['kickSizes'] = paradoxicalKickSizes
            DT.p['kickAmplitudeInh'] = paradoxicalKickAmp
            iKickRecorded = square_bumps(DT.p['kickTimes'],
                                         DT.p['kickDurs'],
                                         DT.p['kickSizes'],
                                         DT.p['duration'],
                                         DT.p['dt'])
            DT.p['iKickRecorded'] = iKickRecorded
            DT.DN.set_paradoxical_kicked()

        DT.run_upCrit()

        R = Results()
        R.init_from_network_object(DT.DN)

        R.calculate_PSTH()
        R.calculate_voltage_histogram(removeMode=True, useAllRecordedUnits=True)
        R.calculate_upstates()
        if len(R.ups) > 0:
            R.reshape_upstates()
            R.calculate_FR_in_upstates()
            print('average FR in upstate for Exc: {:.2f}, Inh: {:.2f} '.format(R.upstateFRExcHist.mean(), R.upstateFRInhHist.mean()))

        R.calculate_voltage_histogram(removeMode=True)
        R.reshape_upstates()

        if pL['paradoxicalKickInh']:  # plot the current injection region

            # print the FR during the paradoxical region vs before
            paraStartTime = paradoxicalKickTimes[0] / second
            paraEndTime = (paradoxicalKickTimes[0] + paradoxicalKickDurs[0]) / second
            paraStartInd = np.argmin(np.abs(R.histCenters - paraStartTime))
            paraEndInd = np.argmin(np.abs(R.histCenters - paraEndTime))

            compStartTime = comparisonStartTime / second
            compEndTime = comparisonEndTime / second
            compStartInd = np.argmin(np.abs(R.histCenters - compStartTime))
            compEndInd = np.argmin(np.abs(R.histCenters - compEndTime))

            paraFRExc = R.FRExc[paraStartInd:paraEndInd].mean()
            paraFRInh = R.FRInh[paraStartInd:paraEndInd].mean()
            compFRExc = R.FRExc[compStartInd:compEndInd].mean()
            compFRInh = R.FRInh[compStartInd:compEndInd].mean()

            paraFRAvgExc[trialInd] = paraFRExc
            paraFRAvgInh[trialInd] = paraFRInh
            compFRAvgExc[trialInd] = compFRExc
            compFRAvgInh[trialInd] = compFRInh

            PSTHExc[currentValueInd, trialInd, :] = R.FRExc
            PSTHInh[currentValueInd, trialInd, :] = R.FRInh

            print('E / I FR during para = {:.2f} / {:.2f}, after para = {:.2f} / {:.2f}'.format(paraFRExc, paraFRInh,
                                                                                                compFRExc, compFRInh))

        del DT
        del R


    paraFRExcsDF = pd.DataFrame(paraFRAvgExc, columns=('FR',))
    paraFRInhsDF = pd.DataFrame(paraFRAvgInh, columns=('FR',))
    compFRExcsDF = pd.DataFrame(compFRAvgExc, columns=('FR',))
    compFRInhsDF = pd.DataFrame(compFRAvgInh, columns=('FR',))

    paraFRExcsDF['timePeriod'] = str(int(paraStartTime)) + '-' + str(int(paraEndTime)) + ' s'
    paraFRInhsDF['timePeriod'] = str(int(paraStartTime)) + '-' + str(int(paraEndTime)) + ' s'
    compFRExcsDF['timePeriod'] = str(int(compStartTime)) + '-' + str(int(compEndTime)) + ' s'
    compFRInhsDF['timePeriod'] = str(int(compStartTime)) + '-' + str(int(compEndTime)) + ' s'

    paraFRExcsDF['unitType'] = 'Ex'
    paraFRInhsDF['unitType'] = 'Inh'
    compFRExcsDF['unitType'] = 'Ex'
    compFRInhsDF['unitType'] = 'Inh'

    frDF = pd.concat((compFRExcsDF, compFRInhsDF, paraFRExcsDF, paraFRInhsDF, ))
    frDF['trialIndex'] = frDF.index
    frDF['currentAmplitude'] = str(currentValue)
    frDF.to_csv(os.path.join(os.getcwd(), p['nameSuffix'] + 'current' + str(currentValueInd) + '_results.csv'))

    if overallDF.size == 0:
        overallDF = frDF.copy()
    else:
        overallDF = pd.concat([overallDF, frDF])

overallDF.to_csv(os.path.join(os.getcwd(), p['nameSuffix'] + '_results.csv'))

savePath = os.path.join(os.getcwd(), p['nameSuffix'] + '_PSTH.npz')
np.savez(savePath, PSTHExc=PSTHExc, PSTHInh=PSTHInh, currentValues=paradoxicalKickAmps)
