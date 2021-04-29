from brian2 import set_device, defaultclock, ms, second, pA, nA, amp, Hz
from params import paramsJercog as p
from params import paramsJercogEphysBuono
from network import JercogNetwork, JercogEphysNetwork
from results import Results, ResultsEphys
import numpy as np
import matplotlib.pyplot as plt
from generate import convert_kicks_to_current_series
from datetime import datetime
import dill
import pickle
import os
from matplotlib.backends.backend_pdf import PdfPages
from generate import adjacency_matrix_from_flat_inds, weight_matrix_from_flat_inds_weights, normal_positive_weights

# for using Brian2GENN
# USE_BRIAN2GENN = False
# if USE_BRIAN2GENN:
#     import brian2genn
#     set_device('genn', debug=False)

defaultclock.dt = p['dt']

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

USE_NEW_EPHYS_PARAMS = True
DISABLE_WEIGHT_SCALING = True

if USE_NEW_EPHYS_PARAMS:
    # remove protected keys from the dict whose params are being imported
    ephysParams = paramsJercogEphysBuono.copy()
    protectedKeys = ('nUnits', 'propInh', 'duration')
    for pK in protectedKeys:
        del ephysParams[pK]
    p.update(ephysParams)

# simulation params
p['simName'] = 'jercogDefault_1e3_P05_eqwts_buono_noscale'
p['nUnits'] = 1e3
p['propConnect'] = 0.5

# define parameters
p['setUpFRExc'] = 5 * Hz
p['setUpFRInh'] = 14 * Hz
p['tauUpFRTrials'] = 2
p['useRule'] = 'balance'  # cross-homeo or balance
p['kickType'] = 'spike'  # kick or spike
p['initWeightMean'] = ''  # default,
p['initWeightDistribution'] = ''  # monolithic, equal,

p['maxAllowedFRExc'] = 25
p['maxAllowedFRInh'] = 60

p['nTrials'] = 1597
p['saveTrials'] = [1, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]  # 1-indexed

p['nUnitsToSpike'] = int(np.round(0.05 * p['nUnits']))
p['timeToSpike'] = 100 * ms
p['timeAfterSpiked'] = 1400 * ms
p['spikeInputAmplitude'] = 0.98 * nA

if p['useRule'] == 'cross-homeo':
    p['alpha1'] = 0.01 * pA / Hz / p['propConnect']  # 0.005
    p['alpha2'] = None
    p['tauPlasticityTrials'] = None
    p['alphaBalance'] = None
    p['minAllowedWEE'] = 1 * pA / p['propConnect']
    p['minAllowedWEI'] = 1 * pA / p['propConnect']
    p['minAllowedWIE'] = 1 * pA / p['propConnect']
    p['minAllowedWII'] = 1 * pA / p['propConnect']
elif p['useRule'] == 'balance':
    # monolithic change version
    # p['alpha1'] = 0.05 * pA * pA / Hz / Hz / Hz / p['propConnect']
    # p['alpha2'] = 0.0005 * pA * pA / Hz / Hz / Hz / p['propConnect']
    # customized change version - no longer multiply by gain (Hz/amp) so must do that here
    p['alpha1'] = 0.05 * pA / p['propConnect']
    p['alpha2'] = 0.005 * pA / p['propConnect']
    p['tauPlasticityTrials'] = 500
    p['alphaBalance'] = 1 / p['tauPlasticityTrials']

    p['minAllowedWEE'] = 1 * pA / p['propConnect']
    p['minAllowedWEI'] = 1 * pA / p['propConnect']
    p['minAllowedWIE'] = 1 * pA / p['propConnect']
    p['minAllowedWII'] = 1 * pA / p['propConnect']

    # p['minAllowedWEI'] = 0.1 * pA / p['propConnect']
    # p['minAllowedWEE'] = p['setUpFRInh'] / p['setUpFRExc'] *\
    #                      p['minAllowedWEI'] + (1 / p['gainExc'] + p['threshExc'] / p['setUpFRExc']) * Hz
    # p['minAllowedWII'] = 0.1 * pA / p['propConnect']
    # p['minAllowedWIE'] = (p['minAllowedWII'] + 1 / p['gainInh'] * Hz) *\
    #                      p['setUpFRInh'] / p['setUpFRExc'] + p['threshInh'] / p['setUpFRExc'] * Hz

# params not important unless using "kick" instead of "spike"
p['propKicked'] = 0.1
p['duration'] = 1500 * ms
p['onlyKickExc'] = True
p['kickTimes'] = [100 * ms]
p['kickSizes'] = [1]
iKickRecorded = convert_kicks_to_current_series(p['kickDur'], p['kickTau'],
                                                p['kickTimes'], p['kickSizes'], p['duration'], p['dt'])
p['iKickRecorded'] = iKickRecorded

p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])
indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)

# first quickly run an EPhys experiment with the given params to calculate the thresh and gain
pForEphys = p.copy()
if 'iExtRange' not in pForEphys:
    pForEphys['propInh'] = 0.5
    pForEphys['duration'] = 250 * ms
    pForEphys['iExtRange'] = np.linspace(0, .3, 31) * nA
JEN = JercogEphysNetwork(pForEphys)
JEN.build_classic()
JEN.run()

RE = ResultsEphys()
RE.init_from_network_object(JEN)
RE.calculate_thresh_and_gain()

p['threshExc'] = RE.threshExc
p['threshInh'] = RE.threshInh
p['gainExc'] = RE.gainExc
p['gainInh'] = RE.gainInh
p['saveTrials'] = np.array(p['saveTrials']) - 1

# set up network, experiment, and start recording
JN = JercogNetwork(p)
JN.initialize_network()

if p['kickType'] == 'kick':
    JN.initialize_units_twice_kickable2()
    JN.set_kicked_units(onlyKickExc=p['onlyKickExc'])
elif p['kickType'] == 'spike':
    JN.initialize_units_twice_kickable2()
    JN.prepare_upCrit_experiment2(minUnits=p['nUnitsToSpike'], maxUnits=p['nUnitsToSpike'],
                                  unitSpacing=5,  # unitSpacing is a useless input in this context
                                  timeSpacing=p['timeAfterSpiked'], startTime=p['timeToSpike'],
                                  currentAmp=p['spikeInputAmplitude'])

JN.initialize_recurrent_synapses_4bundles_modifiable()
JN.create_monitors()

# by multiplying the proposed weight change by the "scale factor"
# we take into account that
# we are modifying a bundle of weights that might be "thicker" pre-synaptically
# and so a small change results in larger total change in the sum of weights onto
# each post-synaptic unit

# let's turn off the scaling to see what happens...
if DISABLE_WEIGHT_SCALING:
    JN.p['wEEScale'] = 1
    JN.p['wIEScale'] = 1
    JN.p['wEIScale'] = 1
    JN.p['wIIScale'] = 1

# initialize history variables
trialUpFRExc = np.empty((p['nTrials'],))
trialUpFRInh = np.empty((p['nTrials'],))
trialUpDur = np.empty((p['nTrials'],))
trialwEE = np.empty((p['nTrials'],))
trialwIE = np.empty((p['nTrials'],))
trialwEI = np.empty((p['nTrials'],))
trialwII = np.empty((p['nTrials'],))

trialUpFRExcUnits = np.empty((p['nTrials'], p['nExc']))
trialUpFRInhUnits = np.empty((p['nTrials'], p['nInh']))

if p['useRule'] == 'cross-homeo':
    # for cross-homeo with weight changes customized to the unit, the same weight change
    # is applied across all incoming presynaptic synapses for each post-synaptic unit
    # so the dW arrays are equal in size to the post-synaptic population
    trialdwEEUnits = np.empty((p['nTrials'], p['nExc']), dtype='float32')
    trialdwEIUnits = np.empty((p['nTrials'], p['nExc']), dtype='float32')
    trialdwIEUnits = np.empty((p['nTrials'], p['nInh']), dtype='float32')
    trialdwIIUnits = np.empty((p['nTrials'], p['nInh']), dtype='float32')
elif p['useRule'] == 'balance':
    trialdwEEUnits = np.empty((p['nTrials'], p['nExc']), dtype='float32')
    trialdwEIUnits = np.empty((p['nTrials'], p['nExc']), dtype='float32')
    trialdwIEUnits = np.empty((p['nTrials'], p['nInh']), dtype='float32')
    trialdwIIUnits = np.empty((p['nTrials'], p['nInh']), dtype='float32')
else:
    trialdwEEUnits = None
    trialdwEIUnits = None
    trialdwIEUnits = None
    trialdwIIUnits = None

# initalize variables to represent the rolling average firing rates of the Exc and Inh units
# we start at None because this is undefined, and we will initialize at the exact value of the first UpFR
movAvgUpFRExc = None
movAvgUpFRInh = None
movAvgUpFRExcUnits = None
movAvgUpFRInhUnits = None

# 'monolithic'
# wEE_init = JN.unitsExc.jE[0]
# wIE_init = JN.unitsInh.jE[0]
# wEI_init = JN.unitsExc.jI[0]
# wII_init = JN.unitsInh.jI[0]

# 'default' and 'equal'
wEE_init = JN.synapsesEE.jEE[:]
wIE_init = JN.synapsesIE.jIE[:]
wEI_init = JN.synapsesEI.jEI[:]
wII_init = JN.synapsesII.jII[:]

# 'default' and 'normal' with SD 0.2 of mean
# wEE_init = JN.synapsesEE.jEE[:] * normal_positive_weights(JN.synapsesEE.jEE[:].size, 1, 0.2)
# wIE_init = JN.synapsesIE.jIE[:] * normal_positive_weights(JN.synapsesIE.jIE[:].size, 1, 0.2)
# wEI_init = JN.synapsesEI.jEI[:] * normal_positive_weights(JN.synapsesEI.jEI[:].size, 1, 0.2)
# wII_init = JN.synapsesII.jII[:] * normal_positive_weights(JN.synapsesII.jII[:].size, 1, 0.2)

# 'default' and 'uniform'
# wEE_init = np.random.rand(JN.synapsesEE.jEE[:].size) * 2 * JN.synapsesEE.jEE[0]
# wIE_init = np.random.rand(JN.synapsesIE.jIE[:].size) * 2 * JN.synapsesIE.jIE[0]
# wEI_init = np.random.rand(JN.synapsesEI.jEI[:].size) * 2 * JN.synapsesEI.jEI[0]
# wII_init = np.random.rand(JN.synapsesII.jII[:].size) * 2 * JN.synapsesII.jII[0]

# 'random' and 'random'
# wEE_init = np.random.rand() * 20 * pA
# wIE_init = np.random.rand() * 20 * pA
# wEI_init = np.random.rand() * 20 * pA
# wII_init = np.random.rand() * 20 * pA

wEE = wEE_init.copy()
wEI = wEI_init.copy()
wIE = wIE_init.copy()
wII = wII_init.copy()

# store the network state
JN.N.store()

# get some adjacency matrix and nPre
aEE = adjacency_matrix_from_flat_inds(p['nExc'], p['nExc'], JN.preEE, JN.posEE)
aIE = adjacency_matrix_from_flat_inds(p['nExc'], p['nInh'], JN.preIE, JN.posIE)
aEI = adjacency_matrix_from_flat_inds(p['nInh'], p['nExc'], JN.preEI, JN.posEI)
aII = adjacency_matrix_from_flat_inds(p['nInh'], p['nInh'], JN.preII, JN.posII)
nPreEE = aEE.sum(0)
nPreEI = aEI.sum(0)
nPreIE = aIE.sum(0)
nPreII = aII.sum(0)

# initialize the pdf
pdfObject = PdfPages(p['saveFolder'] + JN.saveName + '_' + p['useRule'] + '_trials.pdf')

# define message formatters
meanWeightMsgFormatter = ('upstateFRExc: {:.2f} Hz, upstateFRInh: {:.2f}'
                          ' Hz, wEE: {:.2f} pA, wEI: {:.2f} pA, wIE: {:.2f} pA, wII: {:.2f} pA')

figCounter = 1
for trialInd in range(p['nTrials']):

    print('starting trial {}'.format(trialInd + 1))

    # restore the initial network state
    JN.N.restore()

    # set the weights (all weights are equivalent)
    # JN.unitsExc.jE = wEE
    # JN.unitsExc.jI = wEI
    # JN.unitsInh.jE = wIE
    # JN.unitsInh.jI = wII

    # set the weights (separately for each unit)
    JN.synapsesEE.jEE = wEE
    JN.synapsesEI.jEI = wEI
    JN.synapsesIE.jIE = wIE
    JN.synapsesII.jII = wII

    # run the simulation
    t0 = datetime.now()
    JN.run()
    t1 = datetime.now()

    # idea here is that we want to save trials that took a long time so we can figure out why
    if (t1 - t0).seconds > 30:
        saveThisTrial = True
        pickleThisFigure = False  # if you REALLY need to see what's going on, make this True
    else:
        saveThisTrial = trialInd in p['saveTrials']
        pickleThisFigure = False

    # calculate and record the average FR in the up state
    R = Results()
    R.init_from_network_object(JN)
    R.calculate_spike_rate()
    R.calculate_upstates()

    # R.calculate_FR_in_upstates_simply()
    R.calculate_FR_in_upstates_units()  # (separately for each unit)
    print('finished calculating FR in upstates per unit...')

    # save numerical results and/or plots!!!
    if saveThisTrial:
        R.calculate_voltage_histogram(removeMode=True)
        R.reshape_upstates()

        fig1, ax1 = plt.subplots(5, 1, num=figCounter, figsize=(5, 9), gridspec_kw={'height_ratios': [3, 2, 1, 1, 1]},
                                 sharex=True)
        R.plot_spike_raster(ax1[0])
        R.plot_firing_rate(ax1[1])
        ax1[1].set_ylim(0, 30)
        R.plot_voltage_detail(ax1[2], unitType='Exc', useStateInd=0)
        R.plot_updur_lines(ax1[2])
        R.plot_voltage_detail(ax1[3], unitType='Inh', useStateInd=0)
        R.plot_updur_lines(ax1[3])
        R.plot_voltage_detail(ax1[4], unitType='Exc', useStateInd=1)
        R.plot_updur_lines(ax1[4])
        ax1[3].set(xlabel='Time (s)')
        R.plot_voltage_histogram_sideways(ax1[2], 'Exc')
        R.plot_voltage_histogram_sideways(ax1[3], 'Inh')
        fig1.suptitle(R.p['simName'] + '_' + p['useRule'] + '_t' + str(trialInd + 1))
        plt.savefig(pdfObject, format='pdf')
        if pickleThisFigure:
            pickle.dump(fig1,
                        open(
                            R.p['saveFolder'] + '/' + R.rID + '_' + p['useRule'] + '_t' + str(trialInd + 1) + '.pickle',
                            'wb'))

    # if there was not zero Up states
    if len(R.ups) == 1:
        print('there was exactly one Up of duration {:.2f} s'.format(R.upDurs[0]))
        trialUpFRExc[trialInd] = R.upstateFRExc[0]  # in Hz
        trialUpFRInh[trialInd] = R.upstateFRInh[0]  # in Hz
        trialUpFRExcUnits[trialInd, :] = R.upstateFRExcUnits[0, :]  # in Hz
        trialUpFRInhUnits[trialInd, :] = R.upstateFRInhUnits[0, :]  # in Hz
        trialUpDur[trialInd] = R.upDurs[0]
    elif len(R.ups) > 1:
        print('for some reason there were multiple up states!!!')
        break
    else:
        # if there were no Up states, just take the avg FR (near 0 in this case)
        trialUpFRExc[trialInd] = R.FRExc.mean()
        trialUpFRInh[trialInd] = R.FRInh.mean()
        trialUpDur[trialInd] = 0

    # if the currently assessed upstateFR was higher than the saturated FRs of the two types, reduce it
    if trialUpFRExc[trialInd] > p['maxAllowedFRExc']:
        trialUpFRExc[trialInd] = p['maxAllowedFRExc']
    if trialUpFRInh[trialInd] > p['maxAllowedFRInh']:
        trialUpFRInh[trialInd] = p['maxAllowedFRInh']

    # separate by unit, record average weight?
    trialwEE[trialInd] = wEE.mean() / pA
    trialwEI[trialInd] = wEI.mean() / pA
    trialwIE[trialInd] = wIE.mean() / pA
    trialwII[trialInd] = wII.mean() / pA

    # heck it, print those values
    print(meanWeightMsgFormatter.format(trialUpFRExc[trialInd], trialUpFRInh[trialInd], trialwEE[trialInd],
                                        trialwEI[trialInd], trialwIE[trialInd], trialwII[trialInd]))

    # calculate the moving average of the up FRs
    if movAvgUpFRExcUnits is None:

        # movAvgUpFRExc = trialUpFRExc[trialInd] * Hz  # initialize at the first measured
        # movAvgUpFRInh = trialUpFRInh[trialInd] * Hz

        movAvgUpFRExcUnits = trialUpFRExcUnits[trialInd, :] * Hz  # initialize at the first measured
        movAvgUpFRInhUnits = trialUpFRInhUnits[trialInd, :] * Hz

    else:  # this only gets run the first trial (when they are None)

        # movAvgUpFRExc += (-movAvgUpFRExc + trialUpFRExc[trialInd] * Hz) / p['tauUpFRTrials']
        # movAvgUpFRInh += (-movAvgUpFRInh + trialUpFRInh[trialInd] * Hz) / p['tauUpFRTrials']

        movAvgUpFRExcUnits += (-movAvgUpFRExcUnits + trialUpFRExcUnits[trialInd, :] * Hz) / p['tauUpFRTrials']
        movAvgUpFRInhUnits += (-movAvgUpFRInhUnits + trialUpFRInhUnits[trialInd, :] * Hz) / p['tauUpFRTrials']

    if p['useRule'] == 'cross-homeo':

        # dwEE = p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInh)
        # dwEI = -p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInh)
        # dwIE = -p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExc)
        # dwII = p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExc)
        # wEE += dwEE * JN.p['wEEScale']
        # wEI += dwEI * JN.p['wEIScale']
        # wIE += dwIE * JN.p['wIEScale']
        # wII += dwII * JN.p['wIIScale']

        # separately by synapse...
        # must create a vector in which each element represents the average FR
        # of the cells that synapse onto the post
        # for wEI & wEE, mean fr of inh units that target each E unit
        # for wIE & wII, mean fr of exc units that target each I unit
        movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nPreEI
        movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nPreIE

        # check if this is less than 1... if so, make it be 1 Hz
        # movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc < 1 * Hz] = 1 * Hz
        # movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh < 1 * Hz] = 1 * Hz

        # check if this is greater than 2 * set-point, if so, make it be less
        # movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']
        # movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']

        # convert flat weight arrays into matrices in units of pA
        wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
        wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
        wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
        wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

        # proposed weight changes take the form of an array which we interpret as a row vector
        # each element proposes the change to a column of the weight matrix
        # (because the addition broadcasts the row across the columns,
        # the same change is applied to all elements of a column)
        # in other words, all of the incoming synapses to one unit get scaled the same amount
        # depending on the average FR  of the sensor units presynaptic to that unit
        dwEE = p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)
        dwEI = -p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)
        dwIE = -p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)
        dwII = p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)

        # save the proposed weight change in pA
        trialdwEEUnits[trialInd, :] = dwEE / pA
        trialdwEIUnits[trialInd, :] = dwEI / pA
        trialdwIEUnits[trialInd, :] = dwIE / pA
        trialdwIIUnits[trialInd, :] = dwII / pA

        # this broadcasts the addition across the ROWS (the 1d dwEE arrays are row vectors)
        # this applies the same weight change to all incoming synapses onto a single post-synaptic unit
        # but it's a different value for each post-synaptic unit
        wEEMat += dwEE / pA * JN.p['wEEScale']
        wEIMat += dwEI / pA * JN.p['wEIScale']
        wIEMat += dwIE / pA * JN.p['wIEScale']
        wIIMat += dwII / pA * JN.p['wIIScale']

        # reshape back to a matrix
        wEE = wEEMat[JN.preEE, JN.posEE] * pA
        wEI = wEIMat[JN.preEI, JN.posEI] * pA
        wIE = wIEMat[JN.preIE, JN.posIE] * pA
        wII = wIIMat[JN.preII, JN.posII] * pA

    elif p['useRule'] == 'balance':

        # monolithic weight change version
        # dwEE = p['alpha1'] * p['gainExc'] * movAvgUpFRExc * (p['setUpFRExc'] - movAvgUpFRExc)
        # wEE += dwEE * JN.p['wEEScale']
        # dwEI = p['alphaBalance'] * ((p['setUpFRExc'] / p['setUpFRInh'] * wEE * JN.p['wEEScale'] / Hz -
        #                              p['setUpFRExc'] / p['setUpFRInh'] / p['gainExc'] -
        #                              p['threshExc'] / p['setUpFRInh']) - wEI * JN.p['wEIScale'] / Hz)
        # wEI += dwEI * JN.p['wEIScale']
        # dwIE = -p['alpha2'] * p['gainInh'] * movAvgUpFRInh * (p['setUpFRInh'] - movAvgUpFRInh)
        # wIE += dwIE * JN.p['wIEScale']
        # dwII = p['alphaBalance'] * (((wIE * JN.p['wIEScale'] / Hz * p['setUpFRExc'] -
        #                               p['threshInh']) / p['setUpFRInh'] -
        #                              1 / p['gainInh']) - wII * JN.p['wIIScale'] / Hz)
        # wII += dwII * JN.p['wIIScale']

        # customized weight change version

        # start by converting weights to matrices
        # convert flat weight arrays into matrices in units of pA
        wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
        wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
        wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
        wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

        # calculate the average firing rate of exc units that are presynaptic to each exc unit
        movAvgUpFRExcUnitsPreToPostExc = np.matmul(movAvgUpFRExcUnits, aEE) / nPreEE
        # movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nPreIE

        # check if this is less than 1... if so, make it be 1 Hz
        movAvgUpFRExcUnitsPreToPostExc[movAvgUpFRExcUnitsPreToPostExc < 1 * Hz] = 1 * Hz

        # check if this is greater than 2 * set-point, if so, make it be 2 * set-point
        # movAvgUpFRExcUnitsPreToPostExc[movAvgUpFRExcUnitsPreToPostExc > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']

        # take the log to prevent insanely large weight modifications during explosions...
        movAvgUpFRExcUnitsPreToPostExcLog = np.log(movAvgUpFRExcUnitsPreToPostExc / Hz + 1) * Hz

        # weight change is proportional to
        # the error in the post-synaptic FR times the pre-synaptic FR
        # and takes the form of an outer product
        # the first array is simply the error in the post-synaptic FR
        # each element of the second array is the average FR across E units pre-synaptic to that E unit

        # the pre-unit avg FR and post-unit FR errors are both vectors
        # we take the outer product with the pre-unit avg first (column)
        # times the post-unit error second (row)
        # our weight mats are formatted (pre, post) so this works...
        # i.e. each element represents how much to change that weight

        # when we do np.outer, the units fall off (would have been in Hz^2)
        # alpha says how much to change the weight in pA / Hz / Hz

        # (simpler version...)
        # dwEE = p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnits)
        # wEEMat += dwEE / pA * JN.p['wEEScale']

        # (complex version...)
        dwEEMat = p['alpha1'] * np.outer(movAvgUpFRExcUnitsPreToPostExc, (p['setUpFRExc'] - movAvgUpFRExcUnits))
        wEEMat += dwEEMat / pA * JN.p['wEEScale']

        wEE = wEEMat[JN.preEE, JN.posEE] * pA

        # TODO: THIS DOESN'T MAKE ANY FUCKING SENSE

        # (simpler version...)
        # dwIE = p['alpha2'] * (p['setUpFRInh'] - movAvgUpFRInhUnits)
        # wIEMat += dwIE / pA * JN.p['wIEScale']

        # (complex version...)
        dwIEMat = p['alpha2'] * np.outer(movAvgUpFRExcUnitsPreToPostExc, (p['setUpFRInh'] - movAvgUpFRInhUnits))
        wIEMat += dwIEMat / pA * JN.p['wIEScale']

        wIE = wIEMat[JN.preIE, JN.posIE] * pA

        # TODO: the below is fucked up...
        # given the total excitatory input to each E unit,
        # there is an exact amount of inhibition that will result in the the desired relation
        # between the setpoint FRs in the "steady state" (i.e. given that external input
        # is resulting in the desired setpoint FRs)

        # given that we thus know how much inhibitory current is required for each post cell
        # we should modify its incoming inhibitory weights so that they split that equally

        sumExcInputToExc = np.nansum(wEEMat, 0) * pA
        sumInhInputToExc = p['setUpFRExc'] / p['setUpFRInh'] * sumExcInputToExc - \
                           p['setUpFRExc'] / p['setUpFRInh'] / p['gainExc'] / second - \
                           p['threshExc'] / p['setUpFRInh'] / second  # amp

        normwEIMat = wEIMat / np.nansum(wEIMat, 0)  # unitless (amp / amp)
        normlinewEI = normwEIMat * sumInhInputToExc  # amp
        dwEIMat = p['alphaBalance'] * (normlinewEI - wEIMat * pA)
        wEIMat += dwEIMat / pA * JN.p['wEIScale']
        wEI = wEIMat[JN.preEI, JN.posEI] * pA

        sumExcInputToExc = np.nansum(wIEMat, 0) * pA
        sumInhInputToInh = p['setUpFRExc'] / p['setUpFRInh'] * sumExcInputToExc - \
                           1 / p['gainInh'] / second - \
                           p['threshInh'] / p['setUpFRInh'] / second

        normwIIMat = wIIMat / np.nansum(wIIMat, 0)
        normlinewII = normwIIMat * sumInhInputToInh
        dwIIMat = p['alphaBalance'] * (normlinewII - wIIMat * pA)
        wIIMat += dwIIMat / pA * JN.p['wIIScale']
        wII = wIIMat[JN.preII, JN.posII] * pA

        # save the proposed weight change in pA
        # (simple version)...
        # trialdwEEUnits[trialInd, :] = dwEE / pA
        # trialdwEIUnits[trialInd, :] = np.nanmean(dwEIMat, 0) / pA
        # trialdwIEUnits[trialInd, :] = dwIE / pA
        # trialdwIIUnits[trialInd, :] = np.nanmean(dwIIMat, 0) / pA

        # (complex version)...
        trialdwEEUnits[trialInd, :] = np.nansum(dwEEMat, 0) / pA
        trialdwEIUnits[trialInd, :] = np.nansum(dwEIMat, 0) / pA
        trialdwIEUnits[trialInd, :] = np.nansum(dwIEMat, 0) / pA
        trialdwIIUnits[trialInd, :] = np.nansum(dwIIMat, 0) / pA

    # if wEE < p['minAllowedWEE']:
    #     wEE += (p['minAllowedWEE'] - wEE)
    #     # wEE -= dwEE
    # if wEI < p['minAllowedWEI']:
    #     wEI += (p['minAllowedWEI'] - wEI)
    #     # wEI -= dwEI
    # if wIE < p['minAllowedWIE']:
    #     wIE += (p['minAllowedWIE'] - wIE)
    #     # wIE -= dwIE
    # if wII < p['minAllowedWII']:
    #     wII += (p['minAllowedWII'] - wII)
    #     # wII -= dwII

    # how to handle when these are arrays??
    wEETooSmall = wEE < p['minAllowedWEE']
    wIETooSmall = wIE < p['minAllowedWIE']
    wEITooSmall = wEI < p['minAllowedWEI']
    wIITooSmall = wII < p['minAllowedWII']
    if wEETooSmall.any():
        wEE[wEETooSmall] = p['minAllowedWEE']
    if wIETooSmall.any():
        wIE[wIETooSmall] = p['minAllowedWIE']
    if wEITooSmall.any():
        wEI[wEITooSmall] = p['minAllowedWEI']
    if wIITooSmall.any():
        wII[wIITooSmall] = p['minAllowedWII']

    # print(
    #     'movAvgUpFRExc: {:.2f} Hz, movAvgUpFRInh: {:.2f} Hz, dwEE: {:.2f} pA, dwEI: {:.2f} pA, dwIE: {:.2f} pA, dwII: {:.2f} pA'.format(
    #         movAvgUpFRExc, movAvgUpFRInh, dwEE * JN.p['wEEScale'] / pA, dwEI * JN.p['wEIScale'] / pA,
    #         dwIE * JN.p['wIEScale'] / pA, dwII * JN.p['wIIScale'] / pA))

    # separately by unit
    # print(
    #     'movAvgUpFRExc: {:.2f} Hz, movAvgUpFRInh: {:.2f} Hz, dwEE: {:.2f} pA, dwEI: {:.2f} pA, dwIE: {:.2f} pA, dwII: {:.2f} pA'.format(
    #         movAvgUpFRExc, movAvgUpFRInh, dwEEMat.mean() * JN.p['wEEScale'] / pA, dwEIMat.mean() * JN.p['wEIScale'] / pA,
    #                                       dwIEMat.mean() * JN.p['wIEScale'] / pA, dwIIMat.mean() * JN.p['wIIScale'] / pA))

    if p['useRule'] == 'cross-homeo':
        print(
            'movAvgUpFRExc: {:.2f} Hz, movAvgUpFRInh: {:.2f} Hz, dwEE: {:.2f} pA, dwEI: {:.2f} pA, dwIE: {:.2f} pA, dwII: {:.2f} pA'.format(
                movAvgUpFRExc, movAvgUpFRInh, dwEE.sum() * JN.p['wEEScale'] / pA, dwEI.sum() * JN.p['wEIScale'] / pA,
                                              dwIE.sum() * JN.p['wIEScale'] / pA, dwII.sum() * JN.p['wIIScale'] / pA))
        print(
            'mean dwEE: {:.2f} pA, dwIE: {:.2f} pA, dwEI: {:.2f} pA, dwII: {:.2f} pA'.format(
                dwEE.mean() * JN.p['wEEScale'] / pA,
                dwIE.mean() * JN.p['wIEScale'] / pA,
                dwEI.mean() * JN.p['wEIScale'] / pA,
                dwII.mean() * JN.p['wIIScale'] / pA))
    elif p['useRule'] == 'balance':
        print(
            'movAvgUpFRExc: {:.2f} Hz, movAvgUpFRInh: {:.2f} Hz, dwEE: {:.2f} pA, dwIE: {:.2f} pA, dwEI: {:.2f} pA, dwII: {:.2f} pA'.format(
                movAvgUpFRExc, movAvgUpFRInh, np.nansum(dwEEMat) * JN.p['wEEScale'] / pA,
                                              np.nansum(dwIEMat) * JN.p['wIEScale'] / pA,
                                              np.nansum(dwEIMat) * JN.p['wEIScale'] / pA,
                                              np.nansum(dwIIMat) * JN.p['wIIScale'] / pA))
        print(
            'mean dwEE: {:.2f} pA, dwIE: {:.2f} pA, dwEI: {:.2f} pA, dwII: {:.2f} pA'.format(
                np.nanmean(dwEEMat) * JN.p['wEEScale'] / pA,
                np.nanmean(dwIEMat) * JN.p['wIEScale'] / pA,
                np.nanmean(dwEIMat) * JN.p['wEIScale'] / pA,
                np.nanmean(dwIIMat) * JN.p['wIIScale'] / pA))

# close pdf
pdfObject.close()

# params
savePath = os.path.join(p['saveFolder'], JN.saveName + '_' + p['useRule'] + '_params.pkl')
with open(savePath, 'wb') as f:
    dill.dump(p, f)

# results
savePath = os.path.join(JN.p['saveFolder'], JN.saveName + '_' + p['useRule'] + '_results.npz')

np.savez(savePath, trialUpFRExc=trialUpFRExc, trialUpFRInh=trialUpFRInh, trialUpDur=trialUpDur,
         trialwEE=trialwEE, trialwEI=trialwEI, trialwIE=trialwIE, trialwII=trialwII,
         trialdwEEUnits=trialdwEEUnits, trialdwEIUnits=trialdwEIUnits,
         trialdwIEUnits=trialdwIEUnits, trialdwIIUnits=trialdwIIUnits,
         trialUpFRExcUnits=trialUpFRExcUnits, trialUpFRInhUnits=trialUpFRInhUnits,
         wEE_init=wEE_init, wIE_init=wIE_init, wEI_init=wEI_init, wII_init=wII_init,
         wEE_final=wEE, wIE_final=wIE, wEI_final=wEI, wII_final=wII,
         aEE=aEE, aIE=aIE, aEI=aEI, aII=aII,
         )
