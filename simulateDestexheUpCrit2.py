from brian2 import defaultclock, ms, pA, nA, Hz, seed, mV, second, uS
from params import paramsDestexhe as p
from params import paramsDestexheEphysBuono, paramsDestexheEphysOrig
import numpy as np
from generate import convert_kicks_to_current_series, norm_weights, weight_matrix_from_flat_inds_weights, square_bumps
from trainer import DestexheTrainer
from results import Results
import pandas as pd
import matplotlib.pyplot as plt

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
# paradoxicalKickAmps = np.arange(0, 17, 2) * pA
paradoxicalKickAmps = np.arange(0, 21, 5) * pA
# paradoxicalKickAmps = np.array([0, 8]) * pA
paradoxicalKickAmp = 200 * pA

comparisonStartTime = 2000 * ms
comparisonEndTime = 3000 * ms

p['propConnectFeedforwardProjectionUncorr'] = 0.05  # proportion of feedforward projections that are connected
p['nPoissonUncorrInputUnits'] = 2000
p['nUncorrFeedforwardSynapsesPerUnit'] = int(p['propConnectFeedforwardProjectionUncorr'] * 2000 * (1 - p['propInh']))
p['poissonUncorrInputRate'] = 2 * p['nUncorrFeedforwardSynapsesPerUnit'] * Hz
p['qExcFeedforwardUncorr'] = 0.6 * uS / p['nUncorrFeedforwardSynapsesPerUnit']
p['duration'] = 5.1 * second

p['useRule'] = 'upCrit'
p['nameSuffix'] = 'multiMetaDestexheAI1'
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

pL = p.copy()  # pL stands for p in the loop

# set RNG seeds...
rngSeed = None
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

# calculate the best 2 E and single I units based on smallest STD during Up state
if R.ups.size > 0:
    upStartInd = int(R.ups[0] * second / p['stateVariableDT'])
    upEndInd = int(R.downs[0] * second / p['stateVariableDT'])
    voltArrayExc = (DT.JN.stateMonExc.v / mV)[:, upStartInd:upEndInd]
    voltArrayInh = (DT.JN.stateMonInh.v / mV)[:, upStartInd:upEndInd]
    voltArrayExcStd = voltArrayExc.std(1)
    voltArrayInhStd = voltArrayInh.std(1)
    smoothExcList = np.argsort(voltArrayExcStd)
    smoothInhList = np.argsort(voltArrayInhStd)
else:
    smoothExcList = [0, 1, 2]
    smoothInhList = [0, 1, 2]

fig1, ax1 = plt.subplots(5, 1, num=1, figsize=(6, 9),
                         gridspec_kw={'height_ratios': [3, 2, 1, 1, 1]},
                         sharex=True)
R.plot_spike_raster(ax1[0], downSampleUnits=False)  # uses RNG but with a separate random seed
R.plot_firing_rate(ax1[1])
ax1[1].set_ylim(0, 30)
R.plot_voltage_detail(ax1[2], unitType='Exc', useStateInd=smoothExcList[0])
R.plot_updur_lines(ax1[2])
R.plot_voltage_detail(ax1[3], unitType='Inh', useStateInd=smoothInhList[0])
R.plot_updur_lines(ax1[3])
R.plot_voltage_detail(ax1[4], unitType='Exc', useStateInd=smoothExcList[1])
R.plot_updur_lines(ax1[4])
ax1[3].set(xlabel='Time (s)')
R.plot_voltage_histogram_sideways(ax1[2], 'Exc')
R.plot_voltage_histogram_sideways(ax1[3], 'Inh')
fig1.suptitle(R.p['simName'])
uniqueSpikers = np.unique(R.spikeMonExcI).size
totalSpikes = R.spikeMonExcI.size
if uniqueSpikers > 0:
    print(uniqueSpikers, 'neurons fired an average of', totalSpikes / uniqueSpikers, 'spikes')

if p['paradoxicalKickInh']:  # plot the current injection region

    # adorn plot with rectangular patches that show where the current injection took place
    left = paradoxicalKickTimes[0] / second
    width = paradoxicalKickDurs[0] / second
    for anax1 in ax1:
        anax1_ymin, anax1_ymax = anax1.get_ylim()
        bottom = anax1_ymin
        height = anax1_ymax - anax1_ymin
        rect = plt.Rectangle((left, bottom), width, height,
                             facecolor="blue", alpha=0.1)
        anax1.add_patch(rect)

    # print the FR during the paradoxical region vs before
    paraStartInd = np.argmin(np.abs(R.histCenters - (paradoxicalKickTimes[0] / second)))
    paraEndInd = np.argmin(np.abs(R.histCenters - ((paradoxicalKickTimes[0] + paradoxicalKickDurs[0]) / second)))

    compStartInd = np.argmin(np.abs(R.histCenters - (comparisonStartTime / second)))
    compEndInd = np.argmin(np.abs(R.histCenters - (comparisonEndTime / second)))

    paraFRExc = R.FRExc[paraStartInd:paraEndInd].mean()
    paraFRInh = R.FRInh[paraStartInd:paraEndInd].mean()
    compFRExc = R.FRExc[compStartInd:compEndInd].mean()
    compFRInh = R.FRInh[compStartInd:compEndInd].mean()

    print('E / I FR during para = {:.2f} / {:.2f}, before para = {:.2f} / {:.2f}'.format(paraFRExc, paraFRInh,
                                                                                         compFRExc, compFRInh))
