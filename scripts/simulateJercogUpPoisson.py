from brian2 import defaultclock, ms, pA, nA, Hz, seed, mV, second
from params import paramsJercog as p
from params import paramsJercogEphysBuono
import numpy as np
from generate import convert_kicks_to_current_series, weight_matrix_from_flat_inds_weights, norm_weights
from trainer import JercogTrainer
from results import Results
import matplotlib.pyplot as plt
import plotting
import pandas as pd

p['useNewEphysParams'] = True
ephysParams = paramsJercogEphysBuono.copy()
rngSeed = 42
defaultclock.dt = p['dt']

p['useSecondPopExc'] = False

p['manipulateConnectivity'] = True
propEx2Units = 0.1
p['removePropConn'] = 0.1
p['compensateInhibition'] = False
weightMult = 0.85  # 0.85
overrideBetaAdaptExc = 12 * nA * ms  # 13

p['addBackRemovedConns'] = False
p['addBackRemovedConnsE2E2'] = False
p['randomizeConnectivity'] = False

if p['useNewEphysParams']:
    # remove protected keys from the dict whose params are being imported
    protectedKeys = ('nUnits', 'propInh', 'duration')
    for pK in protectedKeys:
        del ephysParams[pK]
    p.update(ephysParams)
yOffset = -72.6

p['useRule'] = 'upPoisson'
p['nameSuffix'] = 'actualPoisson'
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

# good ones
# buonoEphys_2000_0p25_upPoisson_guessLowWeights2e3p025LogNormal2_test_2021-08-13-15-58_results
# buonoEphys_2000_0p25_upPoisson_guessBuono2Weights2e3p025LogNormal_test_2021-08-13-16-06_results
# buonoEphys_2000_0p25_upPoisson_guessBuono2Weights2e3p025LogNormal_test_2021-08-13-16-09_results

# simulation params
p['nUnits'] = 2e3
p['propConnect'] = 0.25
p['allowAutapses'] = False

# p['initWeightMethod'] = 'guessGoodWeights2e3p025LogNormal'
# p['initWeightMethod'] = 'guessBuono7Weights2e3p025'
# p['initWeightMethod'] = 'guessLowWeights2e3p025LogNormal2'
p['initWeightMethod'] = 'resumePrior'
p['initWeightPrior'] = 'buonoEphysBen1_2000_0p25_cross-homeo-pre-outer-homeo_guessBuono7Weights2e3p025SlightLow__2021-09-04-08-20_results'
# p['initWeightPrior'] = 'classicJercog_2000_0p25_cross-homeo-pre-scalar-homeo_goodCrossHomeoExamp__2022-02-16-14-52-13_results'

p['kickType'] = 'spike'  # kick or spike
p['nUnitsToSpike'] = int(np.round(0.05 * p['nUnits']))
p['nUnitsSecondPopExc'] = int(np.round(propEx2Units * p['nUnits']))
p['startIndSecondPopExc'] = p['nUnitsToSpike']

# Poisson
p['poissonLambda'] = 0.2 * Hz  # 0.2
p['duration'] = 60 * second  # 60

# E subpop

# upCrit
p['timeToSpike'] = 100 * ms
p['timeAfterSpiked'] = 3000 * ms
p['spikeInputAmplitude'] = 0.96

# params not important unless using "kick" instead of "spike"
p['propKicked'] = 0.1
p['onlyKickExc'] = True
p['kickTimes'] = [100 * ms]
p['kickSizes'] = [1]
iKickRecorded = convert_kicks_to_current_series(p['kickDur'], p['kickTau'],
                                                p['kickTimes'], p['kickSizes'], p['duration'], p['dt'])
p['iKickRecorded'] = iKickRecorded

# boring params
p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])
indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
indSecondPopExc = p['nUnitsToSpike']
p['indsRecordStateExc'].append(indUnkickedExc)
p['indsRecordStateExc'].append(indSecondPopExc)
p['indsRecordStateExc'].append(1422)  # some random units
p['indsRecordStateExc'].append(87)
p['indsRecordStateExc'].append(1357)
p['indsRecordStateInh'].append(238)

p['nExc'] = int(p['nUnits'] * (1 - p['propInh']))
p['nInh'] = int(p['nUnits'] * p['propInh'])
if p['recordAllVoltage']:
    p['indsRecordStateExc'] = list(range(p['nExc']))
    p['indsRecordStateInh'] = list(range(p['nInh']))

if 'stateVariableDT' not in p:
    p['stateVariableDT'] = p['dt'].copy()

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

    PR.wEE_final = p['rng'].permutation(PR.wEE_final)
    PR.wEI_final = p['rng'].permutation(PR.wEI_final)
    PR.wIE_final = p['rng'].permutation(PR.wIE_final)
    PR.wII_final = p['rng'].permutation(PR.wII_final)

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
    wEEFinal = weight_matrix_from_flat_inds_weights(PR.p['nExc'], PR.p['nExc'], PR.preEE, PR.posEE, PR.wEE_final)

    if p['compensateInhibition']:
        # idea here is that by deleting excitatory connections only, we are upsetting the E/I balance
        # we can calculate the initial E/I balance (basically, summing the rows of wEE and also summing the rows of wEI)
        # then we can compare it to the E/I balance afterward
        wEIInit = weight_matrix_from_flat_inds_weights(PR.p['nInh'], PR.p['nExc'], PR.preEI, PR.posEI, PR.wEI_final)

        # turn the final into a matrix also...
        sumExcOntoExcInit = np.nansum(wEEInit, 0)
        sumInhOntoExcInit = np.nansum(wEIInit, 0)

        sumExcOntoExcFinal = np.nansum(wEEFinal, 0)

        # calculate a ratio of the change
        changeInExcAmount = sumExcOntoExcFinal / sumExcOntoExcInit
        changeInEx2 = changeInExcAmount[startIndExc2:endIndExc2].mean()
        changeInEx1 = changeInExcAmount[endIndExc2:].mean()

        # decrease inhibition appropriately
        wEICompensate = wEIInit.copy()
        wEICompensate[:, startIndExc2:endIndExc2] = wEICompensate[:, startIndExc2:endIndExc2] * changeInEx2
        wEICompensate[:, endIndExc2:] = wEICompensate[:, endIndExc2:] * changeInEx1
        PR.wEI_final = wEICompensate[PR.preEI, PR.posEI]

    if p['addBackRemovedConns']:
        # construct a probability array that is the shape of E2E2, fill it with the value of 1 / (nExc2*nExc2 - nExistingConns - nExc2)
        # in positions where there are not already connections... set existing connection and the diagonal to 0 (should sum to 1)
        # and same for E1E1
        # this will allow us to choose some number of new synapses to add, where there are not already connections, and not on the diag


        nConnRemoved = removeInds.size
        # propAddedConnToE2E2 = p['nUnitsSecondPopExc'] / (p['nExc'] - p['nUnitsToSpike'])
        propAddedConnToE2E2 = (p['nUnitsSecondPopExc'] / (p['nExc'] - p['nUnitsToSpike'])) ** 2  # arguably should be this

        nConnAddedToE2E2 = int(np.round(propAddedConnToE2E2 * nConnRemoved))
        nConnAddedToE1E1 = nConnRemoved - nConnAddedToE2E2

        nExc1 = p['nExc'] - p['nUnitsSecondPopExc'] - p['nUnitsToSpike']
        nExc2 = p['nUnitsSecondPopExc']

        E1E1Inds = np.where(np.logical_and(PR.preEE >= endIndExc2, PR.posEE >= endIndExc2))[0]
        probabilityArrayE1E1 = np.full((nExc1, nExc1), 1 / (nExc1 * nExc1 - E1E1Inds.size - nExc1))
        probabilityArrayE1E1[
            PR.preEE[E1E1Inds] - nExc2 - p['nUnitsToSpike'], PR.posEE[E1E1Inds] - nExc2 - p['nUnitsToSpike']] = 0
        probabilityArrayE1E1[np.diag_indices_from(probabilityArrayE1E1)] = 0
        indicesE1E1Flat = p['rng'].choice(nExc1 ** 2, nConnAddedToE1E1, replace=False, p=probabilityArrayE1E1.ravel())

        E2E2Inds = np.where(np.logical_and(np.logical_and(PR.preEE >= startIndExc2, PR.preEE < endIndExc2),
                                           np.logical_and(PR.posEE >= startIndExc2, PR.posEE < endIndExc2)))[0]
        probabilityArrayE2E2 = np.full((nExc2, nExc2), 1 / (nExc2 * nExc2 - E2E2Inds.size - nExc2))
        probabilityArrayE2E2[PR.preEE[E2E2Inds] - p['nUnitsToSpike'], PR.posEE[E2E2Inds] - p['nUnitsToSpike']] = 0
        probabilityArrayE2E2[np.diag_indices_from(probabilityArrayE2E2)] = 0

        indicesE2E2Flat = p['rng'].choice(nExc2 ** 2, nConnAddedToE2E2, replace=False, p=probabilityArrayE2E2.ravel())

        propConnE2E1 = (E2E1Inds.size - removeE2E1Inds.size) / (nExc2 * nExc1)
        propConnE1E2 = (E1E2Inds.size - removeE1E2Inds.size) / (nExc2 * nExc1)
        propConnE2E2 = (E2E2Inds.size + nConnAddedToE2E2) / (nExc2 * nExc2)
        propConnE1E1 = (E1E1Inds.size + nConnAddedToE1E1) / (nExc1 * nExc1)

        print('E2E1 conn changed from {:.4f} to {:.4f}'.format(E2E1Inds.size / (nExc2 * nExc1), propConnE2E1))
        print('E1E2 conn changed from {:.4f} to {:.4f}'.format(E1E2Inds.size / (nExc2 * nExc1), propConnE1E2))
        print('E2E2 conn changed from {:.4f} to {:.4f}'.format(E2E2Inds.size / (nExc2 * nExc2), propConnE2E2))
        print('E1E1 conn changed from {:.4f} to {:.4f}'.format(E1E1Inds.size / (nExc1 * nExc1), propConnE1E1))

        preIndsE1E1, posIndsE1E1 = np.unravel_index(indicesE1E1Flat, (nExc1, nExc1))
        preIndsE2E2, posIndsE2E2 = np.unravel_index(indicesE2E2Flat, (nExc2, nExc2))

        # must add  + p['nUnitsToSpike'] for E2E2 and  + p['nUnitsToSpike'] + nExc2 for E1E1
        if p['addBackRemovedConnsE2E2']:
            PR.preEE = np.concatenate((PR.preEE, preIndsE2E2 + p['nUnitsToSpike']))
            PR.posEE = np.concatenate((PR.posEE, posIndsE2E2 + p['nUnitsToSpike']))
            PR.wEE_final = np.concatenate((PR.wEE_final, weightsSaved[:preIndsE2E2.size]))

        PR.preEE = np.concatenate((PR.preEE, preIndsE1E1 + p['nUnitsToSpike'] + nExc2))
        PR.posEE = np.concatenate((PR.posEE, posIndsE1E1 + p['nUnitsToSpike'] + nExc2))
        PR.wEE_final = np.concatenate((PR.wEE_final, weightsSaved[preIndsE2E2.size:]))

        if p['compensateInhibition']:
            # turn the final into a matrix also...
            wEEFinal = weight_matrix_from_flat_inds_weights(PR.p['nExc'], PR.p['nExc'], PR.preEE, PR.posEE,
                                                            PR.wEE_final)

            # make inhibition stronger/weaker to compensate
            wEICompensate = weight_matrix_from_flat_inds_weights(PR.p['nInh'], PR.p['nExc'], PR.preEI, PR.posEI,
                                                                 PR.wEI_final)
            wEICompensate[:, endIndExc2:] = wEICompensate[:, endIndExc2:] * propConnE1E1 / PR.p['propConnect']

            # decreaseInhOntoE2Factor = np.nansum(wEEFinal[startIndExc2:endIndExc2, :], 0).mean() / np.nansum(wEEInit[startIndExc2:endIndExc2, :], 0).mean()
            decreaseInhOntoE2Factor = np.nansum(wEEFinal[:, startIndExc2:endIndExc2], 1).mean() / np.nansum(wEEInit[:, startIndExc2:endIndExc2], 1).mean()
            wEICompensate[:, startIndExc2:endIndExc2] = wEICompensate[:, startIndExc2:endIndExc2] * decreaseInhOntoE2Factor
            PR.wEI_final = wEICompensate[PR.preEI, PR.posEI]

PR.p['betaAdaptExc'] = overrideBetaAdaptExc  # override...
JT.set_up_network_Poisson(priorResults=PR, recordAllVoltage=p['recordAllVoltage'])
JT.initialize_weight_matrices()

# manipulate the weights to make things less stable
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
R.calculate_voltage_histogram(removeMode=True, removeReset=True, useAllRecordedUnits=True)
R.calculate_upstates()
R.calculate_upFR_units()
R.calculate_upCorr_units()

if len(R.ups) > 0:
    # R.reshape_upstates()
    R.calculate_FR_in_upstates()
    infoStr = 'average FR in upstate for Exc: {:.2f}, Inh: {:.2f}, average upDur: {:.2f}, upFreq: {:.2f}'.format(
        R.upstateFRExcHist.mean(), R.upstateFRInhHist.mean(), R.upDurs.mean(), R.ups.size / R.p['duration'] / Hz)
    print(infoStr)

# calculate the best 2 E and single I units based on smallest STD during Up state
if R.ups.size > 0:
    checkDT = R.p['duration'] / R.stateMonExcV.shape[1]
    useInwardBy = 50 * ms
    takeTimeInds = []
    for upstateInd in range(len(R.ups)):
        startInd = int((R.ups[upstateInd] * second + useInwardBy) / checkDT)
        endInd = int((R.downs[upstateInd] * second - useInwardBy) / checkDT)
        takeTimeInds.extend(list(range(startInd, endInd)))

    # now cut out
    voltArrayExc = R.stateMonExcV[:, takeTimeInds]
    voltArrayInh = R.stateMonInhV[:, takeTimeInds]
    voltArrayExcStd = voltArrayExc.std(1)
    voltArrayInhStd = voltArrayInh.std(1)
    smoothExcList = np.argsort(voltArrayExcStd)
    smoothInhList = np.argsort(voltArrayInhStd)
    if p['manipulateConnectivity']:
        smoothEx2List = smoothExcList[np.where(np.logical_and(smoothExcList >= startIndExc2, smoothExcList < endIndExc2))]
else:
    smoothExcList = [0, 1, 2]
    smoothInhList = [0, 1, 2]

fig1, ax1 = plt.subplots(5, 1, num=1, figsize=(16, 9),
                         gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]},
                         sharex=True)
R.plot_spike_raster(ax1[0])  # uses RNG but with a separate random seed
R.plot_firing_rate(ax1[1])
ax1[1].set_ylim(0, 30)
R.plot_voltage_detail(ax1[2], unitType='Exc', useStateInd=smoothExcList[0], yOffset=yOffset)
R.plot_voltage_detail(ax1[3], unitType='Inh', useStateInd=smoothInhList[0], yOffset=yOffset)
if p['manipulateConnectivity']:
    R.plot_voltage_detail(ax1[4], unitType='Exc', useStateInd=smoothEx2List[0], yOffset=yOffset,
                          overrideColor='royalblue')
else:
    R.plot_voltage_detail(ax1[4], unitType='Exc', useStateInd=smoothExcList[1], yOffset=yOffset)
R.plot_updur_lines(ax1[2])
R.plot_updur_lines(ax1[3])
R.plot_updur_lines(ax1[4])
ax1[4].set(xlabel='Time (s)')
R.plot_voltage_histogram_sideways(ax1[2], 'Exc')
R.plot_voltage_histogram_sideways(ax1[3], 'Inh')
R.plot_voltage_histogram_sideways(ax1[4], 'Exc')
fig1.suptitle(R.p['simName'])

if p['manipulateConnectivity']:
    # plot the average voltage for Ex1 and Ex2
    f, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    timeVoltage = np.arange(0, R.p['duration'], R.p['stateVariableDT'])
    Ex1Avg = R.stateMonExcV[endIndExc2:, :].mean(0)
    Ex2Avg = R.stateMonExcV[startIndExc2:endIndExc2, :].mean(0)
    ax[0].plot(timeVoltage, Ex1Avg, label='Ex1', color='cyan')
    ax[1].plot(timeVoltage, Ex2Avg, label='Ex2', color='royalblue')
    ax[1].legend()

    # plot the correlation matrix, fuck it
    rhoUpExc = R.rhoUpExc.copy()
    rhoUpExc[np.diag_indices_from(rhoUpExc)] = np.nan
    f, ax = plt.subplots()
    plotting.weight_matrix(ax, rhoUpExc)

if hasattr(R, 'upstateFRExcUnits'):
    frInp = R.upstateFRExcUnits[:, :R.p['nUnitsToSpike']].mean(0)  # input pop
    frEx2 = R.upstateFRExcUnits[:, R.p['nUnitsToSpike']:(R.p['nUnitsToSpike'] + R.p['nUnitsSecondPopExc'])].mean(0)  # secondary pop
    frEx1 = R.upstateFRExcUnits[:, (R.p['nUnitsToSpike'] + R.p['nUnitsSecondPopExc']):].mean(0)  # normal pop
    frInh = R.upstateFRInhUnits.mean(0)

    frInpHat = frInp.mean()
    frEx2Hat = frEx2.mean()
    frEx1Hat = frEx1.mean()
    frInhHat = frInh.mean()

    print(frInpHat, frEx2Hat, frEx1Hat, frInhHat, )
# fig2, ax2 = plt.subplots(8, 1, num=2, figsize=(16, 11), sharex=True)
# R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0, yOffset=yOffset)
# R.plot_voltage_detail(ax2[1], unitType='Exc', useStateInd=1, yOffset=yOffset)
# R.plot_voltage_detail(ax2[2], unitType='Exc', useStateInd=2, yOffset=yOffset)
# R.plot_voltage_detail(ax2[3], unitType='Exc', useStateInd=3, yOffset=yOffset)
# R.plot_voltage_detail(ax2[4], unitType='Exc', useStateInd=4, yOffset=yOffset)
# R.plot_voltage_detail(ax2[5], unitType='Exc', useStateInd=5, yOffset=yOffset)
# R.plot_voltage_detail(ax2[6], unitType='Inh', useStateInd=0, yOffset=yOffset)
# R.plot_voltage_detail(ax2[7], unitType='Inh', useStateInd=1, yOffset=yOffset)
# ax2[6].set(xlabel='Time (s)')
# R.plot_voltage_histogram_sideways(ax2[0], 'Exc')
# R.plot_voltage_histogram_sideways(ax2[7], 'Inh')
# fig1.suptitle(R.p['simName'])

'''
plt.close('all')
# fig1, ax1 = plt.subplots(2, 1, num=1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
fig1, ax1 = plt.subplots(2, 1, num=1, figsize=(5, 5), sharex=True)
R.plot_spike_raster(ax1[0], downSampleUnits=True)
R.plot_firing_rate(ax1[1])
fig1.savefig(targetPath + targetFile + '_spikes.tif')
'''
