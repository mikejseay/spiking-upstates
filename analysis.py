'''
analysing the results, particularly of the trainer
'''

from brian2 import pA, second, Hz
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

from stats import moving_average, regress_linear, remove_outliers
from generate import weight_matrix_from_flat_inds_weights
from plot import weight_matrix
from results import Results


def combine_two_sessions(R1, R2):
    # R will concatenate the "trials" dimension of R1 and R2 for all "trial-based" attributes...
    # while inheriting all the rest of the attributes from R1

    R = Results()
    for attribute in dir(R1):
        if attribute[0] == '_':
            continue
        if isinstance(getattr(R1, attribute), np.ndarray):
            a = getattr(R1, attribute)
            if R1.p['nTrials'] in a.shape:  # this is hacky and will fail when the number of trials matches something
                useAxis = a.shape.index(R1.p['nTrials'])
                setattr(R, attribute, np.concatenate((getattr(R1, attribute), getattr(R2, attribute),), axis=useAxis))
                continue
        setattr(R, attribute, getattr(R1, attribute))
    return R


def generate_description(R):
    if R.p['useRule'][:5] == 'cross':
        R.p['alpha2'] = -1e-12
        R.p['tauPlasticityTrials'] = None
    elif R.p['useRule'] == 'homeo':
        R.p['alpha2'] = -1e-12
        R.p['tauPlasticityTrials'] = None

    if 'wEEScale' not in R.p:
        R.p['wEEScale'] = None
        R.p['wIEScale'] = None
        R.p['wEIScale'] = None
        R.p['wIIScale'] = None

    R.importantInfoString = 'Name: {}\nEE: {}, IE: {}, EI: {}, II: {}\n tauFR={}, a1={:.4f} pA, a2={:.4f} pA, tauP={}'.format(
        R.rID,
        R.p['wEEScale'], R.p['wIEScale'], R.p['wEIScale'], R.p['wIIScale'],
        R.p['tauUpFRTrials'], R.p['alpha1'] / pA, R.p['alpha2'] / pA, R.p['tauPlasticityTrials'])

    R.importantInfoString = R.importantInfoString


def FR_weights(R, startTrialInd=0, endTrialInd=-1):
    # FR and weights... certain trials

    f, ax = plt.subplots(2, 1, sharex=True, figsize=(5, 9))

    ax[0].plot(R.trialUpFRExc[startTrialInd:endTrialInd], label='E', color='g')
    ax[0].plot(R.trialUpFRInh[startTrialInd:endTrialInd], label='I', color='r', alpha=.5)
    ax[0].legend()
    ax[0].hlines(R.p['setUpFRExc'], 0, len(R.trialUpFRExc[startTrialInd:endTrialInd]), ls='--', color='g')
    ax[0].hlines(R.p['setUpFRInh'], 0, len(R.trialUpFRInh[startTrialInd:endTrialInd]), ls='--', color='r')
    # ax[0].set_xlabel('Trial #')
    ax[0].set_ylabel('Firing Rate (Hz)')
    ax[0].set_ylim(0, 20)

    ax[1].plot(R.trialwEE[startTrialInd:endTrialInd], label='wEE', color='cyan')
    ax[1].plot(R.trialwIE[startTrialInd:endTrialInd], label='wIE', color='purple')
    ax[1].plot(R.trialwEI[startTrialInd:endTrialInd], label='wEI', color='cyan', ls='--')
    ax[1].plot(R.trialwII[startTrialInd:endTrialInd], label='wII', color='purple', ls='--')
    ax[1].legend()
    ax[1].set_xlabel('Trial #')
    ax[1].set_ylabel('Weight (pA)')

    f.suptitle(R.importantInfoString)


def calculate_convergence_index(R, movAvgWidth=31):
    # algorithm for deciding when "FR convergence" took place

    wEEDiffAbsMAvg = np.fabs(moving_average(np.ediff1d(R.trialwEE), movAvgWidth))
    wEIDiffAbsMAvg = np.fabs(moving_average(np.ediff1d(R.trialwEI), movAvgWidth))
    wIEDiffAbsMAvg = np.fabs(moving_average(np.ediff1d(R.trialwIE), movAvgWidth))
    wIIDiffAbsMAvg = np.fabs(moving_average(np.ediff1d(R.trialwII), movAvgWidth))

    wSum = wEEDiffAbsMAvg + wEIDiffAbsMAvg + wIEDiffAbsMAvg + wIIDiffAbsMAvg

    return wSum


def determine_first_convergent_trial(R, movAvgWidth=31, movUnderWidth=51):
    # algorithm for deciding when "FR convergence" took place

    wEEDiffAbsMAvg = np.fabs(moving_average(np.ediff1d(R.trialwEE), movAvgWidth))
    wEIDiffAbsMAvg = np.fabs(moving_average(np.ediff1d(R.trialwEI), movAvgWidth))
    wIEDiffAbsMAvg = np.fabs(moving_average(np.ediff1d(R.trialwIE), movAvgWidth))
    wIIDiffAbsMAvg = np.fabs(moving_average(np.ediff1d(R.trialwII), movAvgWidth))

    wSum = wEEDiffAbsMAvg + wEIDiffAbsMAvg + wIEDiffAbsMAvg + wIIDiffAbsMAvg
    wSumMode = mode(wSum)[0][0]

    belowThreshBool = wSum < (wSum.mean() + wSumMode) / 2
    belowThreshBoolMAvg = moving_average(belowThreshBool, movUnderWidth)
    if np.any(belowThreshBoolMAvg == 1):
        firstConvergentInd = np.argmax(belowThreshBoolMAvg == 1)
    else:
        alpha = 0.05
        nTrials = belowThreshBoolMAvg.size
        thresholdInd = int(np.round((1 - alpha) * nTrials))
        thresholdValue = np.sort(belowThreshBoolMAvg)[thresholdInd]
        firstConvergentInd = np.argmax(belowThreshBoolMAvg >= thresholdValue)

    return firstConvergentInd


def determine_drift_rates(R, startTrialInd=1000, endTrialInd=-1):
    # instead of trialwEE, you can use trialdwEEUnits, trialdwEECHUnits, trialdwEEHUnits and take the mean across units or w.e

    xvals = np.arange(len(R.trialwEE[startTrialInd:endTrialInd]))

    _, _, driftRateEERaw, _, _ = regress_linear(xvals, R.trialwEE[startTrialInd:endTrialInd])
    _, _, driftRateEIRaw, _, _ = regress_linear(xvals, R.trialwEI[startTrialInd:endTrialInd])
    _, _, driftRateIERaw, _, _ = regress_linear(xvals, R.trialwIE[startTrialInd:endTrialInd])
    _, _, driftRateIIRaw, _, _ = regress_linear(xvals, R.trialwII[startTrialInd:endTrialInd])

    R.driftRateEE = driftRateEERaw / (R.p['alpha1'] / second / pA)
    R.driftRateEI = driftRateEIRaw / (R.p['alpha1'] / second / pA)
    R.driftRateIE = driftRateIERaw / (R.p['alpha1'] / second / pA)
    R.driftRateII = driftRateIIRaw / (R.p['alpha1'] / second / pA)


def FR_scatter(R, startTrialInd=0, endTrialInd=-1, removeOutliers=False):
    # goal: scatter plot the average E firing rate vs. I firing rate for each trial

    if removeOutliers:
        sd_thresh = 3
        a = R.trialUpFRExc[startTrialInd:endTrialInd]
        b = R.trialUpFRInh[startTrialInd:endTrialInd]
        c = np.sqrt(a ** 2 + b ** 2)
        c_mean = np.nanmean(c)
        c_std = np.nanstd(c)
        c_outlying = np.logical_or(c > c_mean + sd_thresh * c_std, c < c_mean - sd_thresh * c_std)
        xData = a[~c_outlying]
        yData = b[~c_outlying]
    else:
        xData = R.trialUpFRExc[startTrialInd:endTrialInd]
        yData = R.trialUpFRInh[startTrialInd:endTrialInd]

    zData = np.arange(xData.size) + 2

    f, ax = plt.subplots()
    s = ax.scatter(xData, yData, s=3, c=zData, cmap=plt.cm.viridis, alpha=0.5)
    # ax.plot(R.trialUpFRExc, R.trialUpFRInh, color='k', alpha=0.1)
    ax.hlines(R.p['setUpFRInh'] / Hz, ax.get_xlim()[0], ax.get_xlim()[1], linestyles='dotted')
    ax.vlines(R.p['setUpFRExc'] / Hz, ax.get_ylim()[0], ax.get_ylim()[1], linestyles='dotted')
    ax.set_xlabel('E (Hz)')
    ax.set_ylabel('I (Hz)')

    cb = plt.colorbar(s, ax=ax)
    cb.ax.set_ylabel('Trial Index', rotation=270)


def FR_hist2d(R, startTrialInd=0, endTrialInd=-1, removeOutliers=False):
    # goal: 2d hist of the average E firing rate vs. I firing rate for each trial

    if removeOutliers:
        sd_thresh = 3
        a = R.trialUpFRExc[startTrialInd:endTrialInd]
        b = R.trialUpFRInh[startTrialInd:endTrialInd]
        c = np.sqrt(a ** 2 + b ** 2)
        c_mean = np.nanmean(c)
        c_std = np.nanstd(c)
        c_outlying = np.logical_or(c > c_mean + sd_thresh * c_std, c < c_mean - sd_thresh * c_std)
        xData = a[~c_outlying]
        yData = b[~c_outlying]
    else:
        xData = R.trialUpFRExc[startTrialInd:endTrialInd]
        yData = R.trialUpFRInh[startTrialInd:endTrialInd]

    f, ax = plt.subplots()
    h = ax.hist2d(xData, yData, bins=50)
    # ax.plot(R.trialUpFRExc, R.trialUpFRInh, color='k', alpha=0.1)
    ax.hlines(R.p['setUpFRInh'] / Hz, ax.get_xlim()[0], ax.get_xlim()[1], color='white', linestyles='dotted')
    ax.vlines(R.p['setUpFRExc'] / Hz, ax.get_ylim()[0], ax.get_ylim()[1], color='white', linestyles='dotted')
    ax.set_xlabel('E (Hz)')
    ax.set_ylabel('I (Hz)')

    f.colorbar(h[3], ax=ax)


def FR_hist1d_compare(R, startTrialInd=0, endTrialInd=-1):
    # FRs at time points as a histogram (before and after)

    f, ax = plt.subplots(1, 2, figsize=(9, 5), sharex=True, sharey=True)

    ax[0].hist(R.trialUpFRExcUnits[startTrialInd, :], color='g', histtype='step')
    ax[1].hist(R.trialUpFRExcUnits[endTrialInd, :], color='g', histtype='step')

    ax[0].hist(R.trialUpFRInhUnits[startTrialInd, :], color='r', histtype='step')
    ax[1].hist(R.trialUpFRInhUnits[endTrialInd, :], color='r', histtype='step')

    for anax in ax.ravel():
        anax.set(xlabel='Firing Rate (Hz)', ylabel='# of occurences')

    f.suptitle(R.importantInfoString)


def FR_weights_std(R, startTrialInd=0, endTrialInd=-1):
    # FR with standard deviation of FRs and weights, multiple lines

    f, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 9))

    # FRs on bottom...

    trialUpFRExcUnitsSTD = np.std(R.trialUpFRExcUnits[startTrialInd:endTrialInd], 1)
    trialUpFRInhUnitsSTD = np.std(R.trialUpFRInhUnits[startTrialInd:endTrialInd], 1)

    wEEInit = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nExc'], R.preEE, R.posEE, R.wEE_init)
    wIEInit = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nInh'], R.preIE, R.posIE, R.wIE_init)
    wEIInit = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nExc'], R.preEI, R.posEI, R.wEI_init)
    wIIInit = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nInh'], R.preII, R.posII, R.wII_init)

    wEESumInit = np.nanmean(wEEInit, 0)
    wIESumInit = np.nanmean(wIEInit, 0)
    wEISumInit = np.nanmean(wEIInit, 0)
    wIISumInit = np.nanmean(wIIInit, 0)

    trialwEEUnits = wEESumInit + R.trialdwEEUnits
    trialwIEUnits = wIESumInit + R.trialdwIEUnits
    trialwEIUnits = wEISumInit + R.trialdwEIUnits
    trialwIIUnits = wIISumInit + R.trialdwIIUnits

    trialwEESTD = np.std(trialwEEUnits[startTrialInd:endTrialInd], 1)
    trialwIESTD = np.std(trialwIEUnits[startTrialInd:endTrialInd], 1)
    trialwEISTD = np.std(trialwEIUnits[startTrialInd:endTrialInd], 1)
    trialwIISTD = np.std(trialwIIUnits[startTrialInd:endTrialInd], 1)

    ax[0].plot(R.trialwEE[startTrialInd:endTrialInd], label='wEE', color='darkcyan')
    ax[0].plot(R.trialwIE[startTrialInd:endTrialInd], label='wIE', color='indigo')
    ax[0].plot(R.trialwEI[startTrialInd:endTrialInd], label='wEI', color='darkcyan', ls='--')
    ax[0].plot(R.trialwII[startTrialInd:endTrialInd], label='wII', color='indigo', ls='--')

    ax[0].fill_between(np.arange(R.trialwEE[startTrialInd:endTrialInd].size),
                       R.trialwEE[startTrialInd:endTrialInd] - trialwEESTD,
                       R.trialwEE[startTrialInd:endTrialInd] + trialwEESTD,
                       color='cyan', alpha=0.1, linewidth=0.0)
    ax[0].fill_between(np.arange(R.trialwEI[startTrialInd:endTrialInd].size),
                       R.trialwEI[startTrialInd:endTrialInd] - trialwEISTD,
                       R.trialwEI[startTrialInd:endTrialInd] + trialwEISTD,
                       color='cyan', alpha=0.1, linewidth=0.0)
    ax[0].fill_between(np.arange(R.trialwIE[startTrialInd:endTrialInd].size),
                       R.trialwIE[startTrialInd:endTrialInd] - trialwIESTD,
                       R.trialwIE[startTrialInd:endTrialInd] + trialwIESTD,
                       color='purple', alpha=0.1, linewidth=0.0)
    ax[0].fill_between(np.arange(R.trialwII[startTrialInd:endTrialInd].size),
                       R.trialwII[startTrialInd:endTrialInd] - trialwIISTD,
                       R.trialwII[startTrialInd:endTrialInd] + trialwIISTD,
                       color='purple', alpha=0.1, linewidth=0.0)

    ax[0].legend()
    ax[0].set_ylabel('Weight (pA)')

    ax[1].plot(R.trialUpFRExc[startTrialInd:endTrialInd], color='darkgreen', lw=0.2)
    ax[1].plot(R.trialUpFRInh[startTrialInd:endTrialInd], color='darkred', lw=0.2)

    ax[1].fill_between(np.arange(R.trialUpFRExc[startTrialInd:endTrialInd].size),
                       R.trialUpFRExc[startTrialInd:endTrialInd] - trialUpFRExcUnitsSTD,
                       R.trialUpFRExc[startTrialInd:endTrialInd] + trialUpFRExcUnitsSTD,
                       color='green', alpha=0.1, linewidth=0.0)

    ax[1].fill_between(np.arange(R.trialUpFRInh[startTrialInd:endTrialInd].size),
                       R.trialUpFRInh[startTrialInd:endTrialInd] - trialUpFRInhUnitsSTD,
                       R.trialUpFRInh[startTrialInd:endTrialInd] + trialUpFRInhUnitsSTD,
                       color='red', alpha=0.1, linewidth=0.0)

    ax[1].set(xlabel='Trial #', ylabel='Firing Rate (Hz)', ylim=(0, 30))

    f.suptitle(R.importantInfoString)


def weights_scatter_matrix(R, startTrialInd=0, endTrialInd=-1):
    # it looks like the E weights and I weights are correlated with each other... let's take a look at the wEE vs. wEI

    f, ax = plt.subplots(4, 4, figsize=(10, 10))

    weightMatrices = (R.trialwEE, R.trialwEI, R.trialwIE, R.trialwII)
    weightMatrixLabels = ('wEE', 'wEI', 'wIE', 'wII')

    for rowInd in range(4):
        for colInd in range(4):
            if colInd >= rowInd:
                ax[rowInd, colInd].axis('off')
                continue
            s = ax[rowInd, colInd].scatter(weightMatrices[rowInd][startTrialInd:endTrialInd],
                                           weightMatrices[colInd][startTrialInd:endTrialInd],
                                           s=3, c=np.arange(R.trialwEE[startTrialInd:endTrialInd].size) + 2,
                                           cmap=plt.cm.viridis, alpha=0.5)
            if rowInd == 3:
                ax[rowInd, colInd].set_xlabel(weightMatrixLabels[colInd] + ' (pA)')
            if colInd == 0:
                ax[rowInd, colInd].set_ylabel(weightMatrixLabels[rowInd] + ' (pA)')


def weights_hist1d_compare(R):
    # remove outliers and then get the color limts

    wEEInit = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nExc'], R.preEE, R.posEE, R.wEE_init)
    wIEInit = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nInh'], R.preIE, R.posIE, R.wIE_init)
    wEIInit = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nExc'], R.preEI, R.posEI, R.wEI_init)
    wIIInit = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nInh'], R.preII, R.posII, R.wII_init)

    wEEFinal = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nExc'], R.preEE, R.posEE, R.wEE_final)
    wIEFinal = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nInh'], R.preIE, R.posIE, R.wIE_final)
    wEIFinal = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nExc'], R.preEI, R.posEI, R.wEI_final)
    wIIFinal = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nInh'], R.preII, R.posII, R.wII_final)

    wFullInit = np.block([[wEEInit, wIEInit], [-wEIInit, -wIIInit]])
    wFullFinal = np.block([[wEEFinal, wIEFinal], [-wEIFinal, -wIIFinal]])

    allNumbers = np.concatenate((wFullInit, wFullFinal)).ravel()
    allNumbersClean = remove_outliers(allNumbers)

    vlims = (np.nanmin(allNumbersClean), np.nanmax(allNumbersClean))

    # histogram of weights before and after

    wFullInitPlot = wFullInit.copy()
    wFullFinalPlot = wFullFinal.copy()

    wFullInitPlot[wFullInitPlot < vlims[0]] = np.nan
    wFullInitPlot[wFullInitPlot > vlims[1]] = np.nan

    wFullFinalPlot[wFullFinalPlot < vlims[0]] = np.nan
    wFullFinalPlot[wFullFinalPlot > vlims[1]] = np.nan

    fig8, ax8 = plt.subplots(1, 2, figsize=(17, 8))
    ax8[0].hist(wFullInitPlot.ravel(), bins=40)
    ax8[0].set(xlabel='Weight (pA)', ylabel='# of occurences')
    ax8[1].hist(wFullFinalPlot.ravel(), bins=40)
    ax8[1].set(xlabel='Weight (pA)', ylabel='# of occurences')


def weights_matrix_compare(R):
    # weight matrices before and after

    wEEInit = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nExc'], R.preEE, R.posEE, R.wEE_init)
    wIEInit = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nInh'], R.preIE, R.posIE, R.wIE_init)
    wEIInit = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nExc'], R.preEI, R.posEI, R.wEI_init)
    wIIInit = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nInh'], R.preII, R.posII, R.wII_init)

    wEEFinal = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nExc'], R.preEE, R.posEE, R.wEE_final)
    wIEFinal = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nInh'], R.preIE, R.posIE, R.wIE_final)
    wEIFinal = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nExc'], R.preEI, R.posEI, R.wEI_final)
    wIIFinal = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nInh'], R.preII, R.posII, R.wII_final)

    wFullInit = np.block([[wEEInit, wIEInit], [-wEIInit, -wIIInit]])
    wFullFinal = np.block([[wEEFinal, wIEFinal], [-wEIFinal, -wIIFinal]])

    allNumbers = np.concatenate((wFullInit, wFullFinal)).ravel()
    allNumbersClean = remove_outliers(allNumbers)

    vlims = (np.nanmin(allNumbersClean), np.nanmax(allNumbersClean))

    wFullInitPlot = wFullInit.copy()
    wFullFinalPlot = wFullFinal.copy()

    wFullInitPlot[wFullInitPlot < vlims[0]] = np.nan
    wFullInitPlot[wFullInitPlot > vlims[1]] = np.nan

    wFullFinalPlot[wFullFinalPlot < vlims[0]] = np.nan
    wFullFinalPlot[wFullFinalPlot > vlims[1]] = np.nan

    wFullInitPlot[np.isnan(wFullInitPlot)] = 0
    wFullFinalPlot[np.isnan(wFullFinalPlot)] = 0

    fig7, ax7 = plt.subplots(1, 2, figsize=(17, 8))
    weight_matrix(ax7[0], wFullInitPlot, xlabel='Post Index', ylabel='Pre Index',
                  clabel='Weight (pA)', limsMethod='custom', vlims=vlims)
    weight_matrix(ax7[1], wFullFinalPlot, xlabel='Post Index', ylabel='Pre Index',
                  clabel='Weight (pA)', limsMethod='custom', vlims=vlims)

