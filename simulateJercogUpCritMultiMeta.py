from brian2 import defaultclock, ms, pA, nA, Hz, seed, mV, second
from params import paramsJercog as p
from params import (paramsJercogEphysBuono22, paramsJercogEphysBuono4, paramsJercogEphysBuono5, paramsJercogEphysBuono6,
                    paramsJercogEphysBuono7, paramsJercogEphysBuono7InfUp)
import numpy as np
from generate import convert_kicks_to_current_series, norm_weights, weight_matrix_from_flat_inds_weights, square_bumps
from trainer import JercogTrainer
from results import Results

import pandas as pd
import os

rngSeed = None
defaultclock.dt = p['dt']

p['useNewEphysParams'] = True
ephysParams = paramsJercogEphysBuono7InfUp.copy()
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
# paradoxicalKickAmps = np.arange(0, 17, 2) * pA
# paradoxicalKickAmps = np.arange(0, 21, 5) * pA
# paradoxicalKickAmps = np.array([0, 8]) * pA
# paradoxicalKickAmps = np.arange(0, 81, 16) * pA
# paradoxicalKickAmps = np.arange(0, 201, 20) * pA
# paradoxicalKickAmps = np.arange(-70, 1, 10) * pA
paradoxicalKickAmps = np.arange(-200, 1, 20) * pA
# paradoxicalKickAmps = np.arange(0, 21, 4) * pA

comparisonStartTime = 2000 * ms
comparisonEndTime = 3000 * ms

p['propConnectFeedforwardProjectionUncorr'] = 0.05  # proportion of feedforward projections that are connected
p['nPoissonUncorrInputUnits'] = 2000
p['nUncorrFeedforwardSynapsesPerUnit'] = int(p['propConnectFeedforwardProjectionUncorr'] * 2000 * (1 - p['propInh']))
p['poissonUncorrInputRate'] = 2 * p['nUncorrFeedforwardSynapsesPerUnit'] * Hz
p['poissonUncorrInputAmpExc'] = 70  # pA
p['poissonUncorrInputAmpInh'] = 70  # pA
p['duration'] = 5.1 * second

if p['useNewEphysParams']:
    # remove protected keys from the dict whose params are being imported
    protectedKeys = ('nUnits', 'propInh', 'duration')
    for pK in protectedKeys:
        del ephysParams[pK]
    p.update(ephysParams)

p['useRule'] = 'upCrit'
p['nameSuffix'] = 'multiMetaHypAI1'
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
# p['initWeightMethod'] = 'normalAsynchronousIrregular'
# p['initWeightMethod'] = 'identicalAsynchronousIrregular'
p['initWeightMethod'] = 'resumePrior'
p['initWeightPrior'] = 'buonoEphysBen1_2000_0p25_cross-homeo-pre-outer-homeo_guessBuono7Weights2e3p025SlightLow__2021-09-04-08-20_results'
# p['initWeightPrior'] = 'buonoEphysBen1_2000_0p25_cross-homeo-pre-outer-homeo_resumePrior_guessBuono7Weights2e3p025SlightLow_2021-12-09-09-41-18_results'
p['kickType'] = 'barrage'  # kick or spike or barrage
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

outFolderVerbose = os.getcwd() + '\\results_verbose'
outFolder = os.getcwd() + '\\results'

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

        JT = JercogTrainer(pL)

        if pL['initWeightMethod'] == 'resumePrior':
            PR = Results()
            PR.init_from_file(pL['initWeightPrior'], pL['saveFolder'])
            pL = dict(list(pL.items()) + list(PR.p.items()))
            # pL = PR.p.copy()  # note this completely overwrites all settings above
            pL['nameSuffix'] = pL['initWeightMethod'] + pL['nameSuffix']  # a way to remember what it was...
            if 'seed' in pL['nameSuffix']:  # this will only work for single digit seeds...
                rngSeed = int(pL['nameSuffix'][pL['nameSuffix'].find('seed') + 4])
            pL['initWeightMethod'] = 'resumePrior'  # and then we put this back...
        else:
            PR = None

        JT.set_up_network_upCrit(priorResults=PR, recordAllVoltage=pL['recordAllVoltage'])

        if pL['paradoxicalKickInh']:
            JT.p['propKicked'] = paradoxicalKickProp
            JT.p['kickTimes'] = paradoxicalKickTimes
            JT.p['kickDurs'] = paradoxicalKickDurs
            JT.p['kickSizes'] = paradoxicalKickSizes
            JT.p['kickAmplitudeInh'] = paradoxicalKickAmp
            iKickRecorded = square_bumps(JT.p['kickTimes'],
                                         JT.p['kickDurs'],
                                         JT.p['kickSizes'],
                                         JT.p['duration'],
                                         JT.p['dt'])
            JT.p['iKickRecorded'] = iKickRecorded
            JT.JN.set_paradoxical_kicked()

        JT.initialize_weight_matrices()

        # here i will do so by multiplying by normal noise
        # or by simply decreasing all the weights by a constant multiplier
        JT.wEE_init = JT.wEE_init * weightMult
        JT.wEI_init = JT.wEI_init * weightMult
        JT.wIE_init = JT.wIE_init * weightMult
        JT.wII_init = JT.wII_init * weightMult

        JT.run_upCrit()
        # JT.save_params()
        # JT.save_results_upCrit()

        R = Results()
        # R.init_from_file(JT.saveName, JT.p['saveFolder'])
        R.init_from_network_object(JT.JN)

        R.calculate_PSTH()
        R.calculate_voltage_histogram(removeMode=True, useAllRecordedUnits=True)
        R.calculate_upstates()
        if len(R.ups) > 0:
            # R.reshape_upstates()
            R.calculate_FR_in_upstates()
            print('average FR in upstate for Exc: {:.2f}, Inh: {:.2f} '.format(R.upstateFRExcHist.mean(), R.upstateFRInhHist.mean()))

        R.calculate_voltage_histogram(removeMode=True)
        # R.reshape_upstates()

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

        del JT
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
    frDF.to_csv(os.path.join(outFolderVerbose, p['nameSuffix'] + 'current' + str(currentValueInd) + '_results.csv'))

    if overallDF.size == 0:
        overallDF = frDF.copy()
    else:
        overallDF = pd.concat([overallDF, frDF])

overallDF.to_csv(os.path.join(outFolder, p['nameSuffix'] + '_results.csv'))

savePath = os.path.join(outFolder, p['nameSuffix'] + '_PSTH.npz')
np.savez(savePath, PSTHExc=PSTHExc, PSTHInh=PSTHInh, currentValues=paradoxicalKickAmps)
