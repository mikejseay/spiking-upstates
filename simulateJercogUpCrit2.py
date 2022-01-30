from brian2 import defaultclock, ms, pA, nA, Hz, seed, mV, second
from params import paramsJercog as p
from params import (paramsJercogEphysBuono22, paramsJercogEphysBuono4, paramsJercogEphysBuono5, paramsJercogEphysBuono6,
                    paramsJercogEphysBuono7, paramsJercogEphysBuono7InfUp)
import numpy as np
from generate import convert_kicks_to_current_series, norm_weights, weight_matrix_from_flat_inds_weights, square_bumps
from trainer import JercogTrainer
from results import Results
import matplotlib.pyplot as plt
from analysis import calculate_net_current_units

rngSeed = None
defaultclock.dt = p['dt']

p['useNewEphysParams'] = True
ephysParams = paramsJercogEphysBuono7.copy()
# ephysParams = paramsJercogEphysBuono7InfUp.copy()
p['useSecondPopExc'] = False
p['randomizeConnectivity'] = False
p['manipulateConnectivity'] = False
p['removePropConn'] = 0.2
p['addBackRemovedConns'] = False
p['paradoxicalKickInh'] = False
weightMult = 1

# PARADOXICAL EFFECT EXPERIMENT PARAMETERS
paradoxicalKickProp = 1
paradoxicalKickTimes = [3000 * ms]
paradoxicalKickDurs = [1000 * ms]
paradoxicalKickSizes = [1]
paradoxicalKickAmp = -70 * pA

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
p['nameSuffix'] = 'paradox1'
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
p['kickType'] = 'spike'  # kick or spike or barrage
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

# set RNG seeds...
p['rngSeed'] = rngSeed
rng = np.random.default_rng(rngSeed)  # for numpy
seed(rngSeed)  # for Brian... will insert code to set the random number generator seed into the generated code
p['rng'] = rng

JT = JercogTrainer(p)

if p['initWeightMethod'] == 'resumePrior':
    PR = Results()
    PR.init_from_file(p['initWeightPrior'], p['saveFolder'])
    p = dict(list(p.items()) + list(PR.p.items()))
    # p = PR.p.copy()  # note this completely overwrites all settings above
    p['nameSuffix'] = p['initWeightMethod'] + p['nameSuffix']  # a way to remember what it was...
    if 'seed' in p['nameSuffix']:  # this will only work for single digit seeds...
        rngSeed = int(p['nameSuffix'][p['nameSuffix'].find('seed') + 4])
    p['initWeightMethod'] = 'resumePrior'  # and then we put this back...
else:
    PR = None

if p['randomizeConnectivity']:
    netCurrentExcPre, netCurrentInhPre = calculate_net_current_units(PR)

    PR.wEE_final = p['rng'].permutation(PR.wEE_final)
    PR.wEI_final = p['rng'].permutation(PR.wEI_final)
    PR.wIE_final = p['rng'].permutation(PR.wIE_final)
    PR.wII_final = p['rng'].permutation(PR.wII_final)

    netCurrentExcPost, netCurrentInhPost = calculate_net_current_units(PR)

    f, ax = plt.subplots(2, 1)
    ax[0].hist(netCurrentExcPre, histtype='step', label='original E')
    ax[0].hist(netCurrentExcPost, histtype='step', label='shuffled E')
    ax[0].legend()

    ax[1].hist(netCurrentInhPre, histtype='step', label='original I')
    ax[1].hist(netCurrentInhPost, histtype='step', label='shuffled I')
    ax[1].legend()

if p['manipulateConnectivity']:

    startIndExc2 = p['startIndSecondPopExc']
    endIndExc2 = p['startIndSecondPopExc'] + p['nUnitsSecondPopExc']

    E2E1Inds = \
        np.where(
            np.logical_and(PR.preEE >= endIndExc2, np.logical_and(PR.posEE >= startIndExc2, PR.posEE < endIndExc2)))[0]
    E1E2Inds = \
        np.where(
            np.logical_and(PR.posEE >= endIndExc2, np.logical_and(PR.preEE >= startIndExc2, PR.preEE < endIndExc2)))[0]

    removeE2E1Inds = p['rng'].choice(E2E1Inds, int(np.round(E2E1Inds.size * p['removePropConn'])), replace=False)
    removeE1E2Inds = p['rng'].choice(E1E2Inds, int(np.round(E1E2Inds.size * p['removePropConn'])), replace=False)

    removeInds = np.concatenate((removeE2E1Inds, removeE1E2Inds))

    # save some info...
    wEEInit = weight_matrix_from_flat_inds_weights(PR.p['nExc'], PR.p['nExc'], PR.preEE, PR.posEE, PR.wEE_final)
    weightsSaved = PR.wEE_final[removeInds]  # save the weights to be used below

    PR.preEE = np.delete(PR.preEE, removeInds, None)
    PR.posEE = np.delete(PR.posEE, removeInds, None)
    PR.wEE_final = np.delete(PR.wEE_final, removeInds, None)

    if p['addBackRemovedConns']:
        nConnRemoved = removeInds.size
        propAddedConnToE2E2 = p['nUnitsSecondPopExc'] / (p['nExc'] - p['nUnitsToSpike'])
        # propAddedConnToE2E2 = (p['nUnitsSecondPopExc'] / (p['nExc'] - p['nUnitsToSpike'])) ** 2  # arguably should be this
        nConnAddedToE2E2 = int(np.round(propAddedConnToE2E2 * nConnRemoved))
        nConnAddedToE1E1 = nConnRemoved - nConnAddedToE2E2
        E2E2Inds = np.where(np.logical_and(np.logical_and(PR.preEE >= startIndExc2, PR.preEE < endIndExc2),
                                           np.logical_and(PR.posEE >= startIndExc2, PR.posEE < endIndExc2)))[0]
        E1E1Inds = np.where(np.logical_and(PR.preEE >= endIndExc2, PR.posEE >= endIndExc2))[0]

        # construct a probability array that is the shape of E2E2, fill it with the value of 1 / (nExc2*nExc2 - nExistingConns - nExc2)
        # in positions where there are not already connections... set existing connection and the diagonal to 0 (should sum to 1)
        # and same for E1E1
        # this will allow us to choose some number of new synapses to add, where there are not already connections, and not on the diag

        nExc1 = p['nExc'] - p['nUnitsSecondPopExc'] - p['nUnitsToSpike']
        nExc2 = p['nUnitsSecondPopExc']
        probabilityArrayE2E2 = np.full((nExc2, nExc2), 1 / (nExc2 * nExc2 - E2E2Inds.size - nExc2))
        probabilityArrayE2E2[PR.preEE[E2E2Inds] - p['nUnitsToSpike'], PR.posEE[E2E2Inds] - p['nUnitsToSpike']] = 0
        probabilityArrayE2E2[np.diag_indices_from(probabilityArrayE2E2)] = 0
        probabilityArrayE1E1 = np.full((nExc1, nExc1), 1 / (nExc1 * nExc1 - E1E1Inds.size - nExc1))
        probabilityArrayE1E1[
            PR.preEE[E1E1Inds] - nExc2 - p['nUnitsToSpike'], PR.posEE[E1E1Inds] - nExc2 - p['nUnitsToSpike']] = 0
        probabilityArrayE1E1[np.diag_indices_from(probabilityArrayE1E1)] = 0

        indicesE2E2Flat = p['rng'].choice(nExc2 ** 2, nConnAddedToE2E2, replace=False, p=probabilityArrayE2E2.ravel())
        indicesE1E1Flat = p['rng'].choice(nExc1 ** 2, nConnAddedToE1E1, replace=False, p=probabilityArrayE1E1.ravel())

        propConnE2E1 = (E2E1Inds.size - removeE2E1Inds.size) / (nExc2 * nExc1)
        propConnE1E2 = (E1E2Inds.size - removeE1E2Inds.size) / (nExc2 * nExc1)
        propConnE2E2 = (E2E2Inds.size + nConnAddedToE2E2) / (nExc2 * nExc2)
        propConnE1E1 = (E1E1Inds.size + nConnAddedToE1E1) / (nExc1 * nExc1)

        print(propConnE2E1)
        print(propConnE1E2)
        print(propConnE2E2)
        print(propConnE1E1)

        # must add  + p['nUnitsToSpike'] for E2E2 and  + p['nUnitsToSpike'] + nExc2 for E1E1
        preIndsE2E2, posIndsE2E2 = np.unravel_index(indicesE2E2Flat, (nExc2, nExc2))
        preIndsE1E1, posIndsE1E1 = np.unravel_index(indicesE1E1Flat, (nExc1, nExc1))
        PR.preEE = np.concatenate((PR.preEE, preIndsE2E2 + p['nUnitsToSpike']))
        PR.posEE = np.concatenate((PR.posEE, posIndsE2E2 + p['nUnitsToSpike']))
        PR.wEE_final = np.concatenate((PR.wEE_final, weightsSaved[:preIndsE2E2.size]))
        PR.preEE = np.concatenate((PR.preEE, preIndsE1E1 + p['nUnitsToSpike'] + nExc2))
        PR.posEE = np.concatenate((PR.posEE, posIndsE1E1 + p['nUnitsToSpike'] + nExc2))
        PR.wEE_final = np.concatenate((PR.wEE_final, weightsSaved[preIndsE2E2.size:]))

        # make inhibition stronger/weaker to compensate
        wEEFinal = weight_matrix_from_flat_inds_weights(PR.p['nExc'], PR.p['nExc'], PR.preEE, PR.posEE, PR.wEE_final)
        wEICompensate = weight_matrix_from_flat_inds_weights(PR.p['nInh'], PR.p['nExc'], PR.preEI, PR.posEI,
                                                             PR.wEI_final)
        wEICompensate[:, endIndExc2:] = wEICompensate[:, endIndExc2:] * propConnE1E1 / PR.p['propConnect']

        # decreaseInhOntoE2Factor = np.nansum(wEEFinal[startIndExc2:endIndExc2, :], 0).mean() / np.nansum(wEEInit[startIndExc2:endIndExc2, :], 0).mean()
        decreaseInhOntoE2Factor = np.nansum(wEEFinal[:, startIndExc2:endIndExc2], 1).mean() / np.nansum(wEEInit[:, startIndExc2:endIndExc2], 1).mean()
        wEICompensate[:, startIndExc2:endIndExc2] = wEICompensate[:, startIndExc2:endIndExc2] * decreaseInhOntoE2Factor
        PR.wEI_final = wEICompensate[PR.preEI, PR.posEI]

JT.set_up_network_upCrit(priorResults=PR, recordAllVoltage=p['recordAllVoltage'])

if p['paradoxicalKickInh']:
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

# JT.wEE_init = JT.wEE_init * norm_weights(JT.wEE_init.shape[0])
# JT.wEI_init = JT.wEI_init * norm_weights(JT.wEI_init.shape[0])
# JT.wIE_init = JT.wIE_init * norm_weights(JT.wIE_init.shape[0])
# JT.wII_init = JT.wII_init * norm_weights(JT.wII_init.shape[0])

JT.run_upCrit()
JT.save_params()
JT.save_results_upCrit()

R = Results()
R.init_from_file(JT.saveName, JT.p['saveFolder'])

R.calculate_PSTH()
R.calculate_voltage_histogram(removeMode=True, useAllRecordedUnits=True)
R.calculate_upstates()
if len(R.ups) > 0:
    # R.reshape_upstates()
    R.calculate_FR_in_upstates()
    print('average FR in upstate for Exc: {:.2f}, Inh: {:.2f} '.format(R.upstateFRExcHist.mean(), R.upstateFRInhHist.mean()))


R.calculate_voltage_histogram(removeMode=True)

# calculate the best 2 E and single I units based on smallest STD during Up state
if R.ups.size > 0:
    upStartInd = int(R.ups[0] * second / p['stateVariableDT'])
    upEndInd = int(R.downs[0] * second / p['stateVariableDT'])
    voltArrayExc = (JT.JN.stateMonExc.v / mV)[:, upStartInd:upEndInd]
    voltArrayInh = (JT.JN.stateMonInh.v / mV)[:, upStartInd:upEndInd]
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

    # this is some weird hack i thought might work but it didn't
    useE = compFRExc
    useI = compFRInh
    ESet = 5
    ISet = 14
    dwEE = useE * (ISet - useI + ESet - useE)
    dwEI = -useI * (ISet - useI + ESet - useE)
    dwIE = useE * (ISet - useI - ESet + useE)
    dwII = useI * (ISet - useI + ESet - useE)
    print(dwEE)
    print(dwEI)
    print(dwIE)
    print(dwII)

if p['useSecondPopExc']:
    fig2, ax2 = plt.subplots(6, 1, num=2, figsize=(6, 9),
                             gridspec_kw={'height_ratios': [3, 2, 1, 1, 1, 1]},
                             sharex=True)
    R.plot_spike_raster(ax2[0], downSampleUnits=False)  # uses RNG but with a separate random seed
    R.plot_firing_rate(ax2[1])
    ax2[1].set_ylim(0, 30)
    R.plot_voltage_detail(ax2[2], unitType='Exc', useStateInd=smoothExcList[smoothExcList < p['nUnitsToSpike']][0])
    R.plot_updur_lines(ax2[2])
    R.plot_voltage_detail(ax2[3], unitType='Exc', useStateInd=smoothExcList[0])
    R.plot_updur_lines(ax2[3])
    # smoothExcList[np.logical_and(smoothExcList >= startIndExc2, smoothExcList < endIndExc2)][0]
    R.plot_voltage_detail(ax2[4], unitType='Exc', useStateInd=p['nUnitsSecondPopExc'])
    R.plot_updur_lines(ax2[4])
    R.plot_voltage_detail(ax2[5], unitType='Inh', useStateInd=smoothInhList[0])
    R.plot_updur_lines(ax2[5])
    ax2[5].set(xlabel='Time (s)')
    R.plot_voltage_histogram_sideways(ax2[2], 'Exc')
    R.plot_voltage_histogram_sideways(ax2[5], 'Inh')
    fig1.suptitle(R.p['simName'])
    uniqueSpikers = np.unique(R.spikeMonExcI).size
    totalSpikes = R.spikeMonExcI.size
    print(uniqueSpikers, 'neurons fired an average of', totalSpikes / uniqueSpikers, 'spikes')
