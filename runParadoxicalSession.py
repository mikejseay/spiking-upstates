import os
import numpy as np

from brian2 import defaultclock, ms, pA, seed
from params import paramsJercog as p
from generate import square_bumps
from trainer import JercogTrainer
from results import Results

rngSeed = None
defaultclock.dt = p['dt']

# PARADOXICAL EFFECT EXPERIMENT PARAMETERS
p['kickAmplitudeExc'] = 0 * pA  # only kick the inh units
p['paradoxicalKickInh'] = True
paradoxicalKickProp = 1
paradoxicalKickTimes = [3000 * ms]
paradoxicalKickDurs = [1000 * ms]
paradoxicalKickSizes = [1]
# paradoxicalKickAmps = np.arange(0, 6, 1) * pA
paradoxicalKickAmps = np.arange(0, 7, 2) * pA
nTrials = 40

p['timeToSpike'] = 100 * ms

p['useRule'] = 'upCrit'
p['nameSuffix'] = 'userRunParadox'
p['saveFolder'] = os.path.join(os.getcwd(), 'results')
p['saveWithDate'] = True

p['downSampleVoltageTo'] = 1 * ms
p['stateVariableDT'] = 1 * ms
p['dtHistPSTH'] = 10 * ms

# simulation params
p['nUnits'] = 2e3
p['propConnect'] = 0.25
p['allowAutapses'] = False

p['initWeightMethod'] = 'resumePrior'
# p['initWeightPrior'] = 'classicJercog_2000_0p25_cross-homeo-pre-outer-homeo_goodCrossHomeoExamp__2022-01-27-07-30-31_results'
p['initWeightPrior'] = 'classicJercog_2000_0p25_cross-homeo-pre-outer-homeo_goodCrossHomeoExamp_userRunTraining_2022-05-24-12-42-16_results'

p['kickType'] = 'spike'  # kick or spike or barrage
p['nUnitsToSpike'] = int(np.round(0.05 * p['nUnits']))
p['timeToSpike'] = 100 * ms
p['timeAfterSpiked'] = paradoxicalKickTimes[-1] + paradoxicalKickDurs[-1] + 1000 * ms
p['spikeInputAmplitude'] = 0.98  # in nA

# boring params
p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])
indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)
p['nExc'] = int(p['nUnits'] * (1 - p['propInh']))
p['nInh'] = int(p['nUnits'] * p['propInh'])

nCurrentValues = paradoxicalKickAmps.size
estimatedDuration = paradoxicalKickTimes[-1] + paradoxicalKickDurs[-1] + p['timeToSpike'] + 1000 * ms
estimatedNBinCenters = int(np.round(estimatedDuration / p['dtHistPSTH'])) - 1

PSTHExc = np.empty((nCurrentValues, nTrials, estimatedNBinCenters))
PSTHInh = np.empty((nCurrentValues, nTrials, estimatedNBinCenters))

# END OF PARAMS

# set RNG seeds...
p['rngSeed'] = rngSeed
rng = np.random.default_rng(rngSeed)  # for numpy
seed(rngSeed)  # for Brian... will insert code to set the random number generator seed into the generated code
p['rng'] = rng

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

        JT.set_up_network_upCrit(priorResults=PR)

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
        JT.run_upCrit()

        R = Results()
        R.init_from_network_object(JT.JN)

        R.calculate_PSTH()

        # f, ax = plt.subplots()
        # R.plot_voltage_detail(ax, unitType='Exc', useStateInd=1)
        # R.plot_voltage_detail(ax, unitType='Inh', useStateInd=0)

        if pL['paradoxicalKickInh']:  # plot the current injection region

            PSTHExc[currentValueInd, trialInd, :] = R.FRExc
            PSTHInh[currentValueInd, trialInd, :] = R.FRInh

        del JT
        del R

savePath = os.path.join(p['saveFolder'], p['nameSuffix'] + '_PSTH.npz')
np.savez(savePath, PSTHExc=PSTHExc, PSTHInh=PSTHInh, currentValues=paradoxicalKickAmps)
