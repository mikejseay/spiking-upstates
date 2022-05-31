'''
analysing the results, particularly of the trainer
'''

from brian2 import pA, second, Hz
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

from stats import moving_average, regress_linear, remove_outliers
from generate import weight_matrix_from_flat_inds_weights
from plotting import weight_matrix
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


def FR_weights(R, startTrialInd=0, endTrialInd=-1, movAvgWidth=None):
    # FR and weights... certain trials

    f, ax = plt.subplots(2, 1, sharex=True, figsize=(5, 9))

    if movAvgWidth:
        useFRExc = moving_average(R.trialUpFRExc[startTrialInd:endTrialInd], movAvgWidth)
        useFRInh = moving_average(R.trialUpFRInh[startTrialInd:endTrialInd], movAvgWidth)
    else:
        useFRExc = R.trialUpFRExc[startTrialInd:endTrialInd]
        useFRInh = R.trialUpFRInh[startTrialInd:endTrialInd]

    ax[0].plot(useFRExc, label='E', color='g')
    ax[0].plot(useFRInh, label='I', color='r', alpha=.5)
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

    return f, ax


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

    return f, ax


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

    return f, ax


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

    return f, ax


def FR_hist1d(R, trialIndex=-1, minFR=0, maxFR=30, nBins=40):

    f, ax = plt.subplots(figsize=(7, 5.5))

    frBins = np.linspace(minFR, maxFR, nBins)

    ax.hist(R.trialUpFRExcUnits[trialIndex, :], frBins, color='cyan', density=True, histtype='step', label='E')
    ax.hist(R.trialUpFRInhUnits[trialIndex, :], frBins, color='r', density=True, histtype='step', label='I')
    ax.set_xlabel('FR (Hz)')
    ax.set_ylabel('prop. of occurences')
    ax.legend(frameon=False)

    return f, ax


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

    return f, ax


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

    return f, ax


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

    f, ax = plt.subplots(1, 2, figsize=(17, 8))
    ax[0].hist(wFullInitPlot.ravel(), bins=40)
    ax[0].set(xlabel='Weight (pA)', ylabel='# of occurences')
    ax[1].hist(wFullFinalPlot.ravel(), bins=40)
    ax[1].set(xlabel='Weight (pA)', ylabel='# of occurences')

    return f, ax


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

    f, ax = plt.subplots(1, 2, figsize=(17, 8))
    weight_matrix(ax[0], wFullInitPlot, xlabel='Post Index', ylabel='Pre Index',
                  clabel='Weight (pA)', limsMethod='custom', vlims=vlims)
    weight_matrix(ax[1], wFullFinalPlot, xlabel='Post Index', ylabel='Pre Index',
                  clabel='Weight (pA)', limsMethod='custom', vlims=vlims)

    return f, ax


def summed_weights_scatter(R):

    wEEInit = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nExc'], R.preEE, R.posEE, R.wEE_init)
    wIEInit = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nInh'], R.preIE, R.posIE, R.wIE_init)
    wEIInit = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nExc'], R.preEI, R.posEI, R.wEI_init)
    wIIInit = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nInh'], R.preII, R.posII, R.wII_init)

    wEEFinal = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nExc'], R.preEE, R.posEE, R.wEE_final)
    wIEFinal = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nInh'], R.preIE, R.posIE, R.wIE_final)
    wEIFinal = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nExc'], R.preEI, R.posEI, R.wEI_final)
    wIIFinal = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nInh'], R.preII, R.posII, R.wII_final)

    wEEInitP = np.nansum(wEEInit, 0)
    wIEInitP = np.nansum(wIEInit, 0)
    wEIInitP = np.nansum(wEIInit, 0)
    wIIInitP = np.nansum(wIIInit, 0)

    wEEFinalP = np.nansum(wEEFinal, 0)
    wIEFinalP = np.nansum(wIEFinal, 0)
    wEIFinalP = np.nansum(wEIFinal, 0)
    wIIFinalP = np.nansum(wIIFinal, 0)

    f, axs = plt.subplots(2, 2)

    xValSeq = (wEEInitP, wIEInitP, wEEFinalP, wIEFinalP,)
    yValSeq = (wEIInitP, wIIInitP, wEIFinalP, wIIFinalP,)

    xLblSeq = ('J EE Init (nA)', 'J IE Init (nA)', 'J EE Final (nA)', 'J IE Final (nA)', )
    yLblSeq = ('J EI Init (nA)', 'J II Init (nA)', 'J EI Final (nA)', 'J II Final (nA)', )

    for axInd, ax in enumerate(axs.ravel()):
        xVals = xValSeq[axInd] / 1e3
        yVals = yValSeq[axInd] / 1e3
        ax.scatter(xVals, yVals, 1)
        ax.set_xlabel(xLblSeq[axInd])
        ax.set_ylabel(yLblSeq[axInd])

    return f, axs


def detail_trial_plot(R, useTrialInd=0):

    actualTrialIndex = R.p['saveTrials'][useTrialInd] + 1
    frameMult = int(R.p['downSampleVoltageTo'] / R.p['dt'])
    useUnitInd = 0
    decreaseVoltageBy = 70
    FRDT = 0.01  # seconds
    voltageYLims = (-90, 10)
    targetDisplayedExcUnits = 160
    targetDisplayedInhUnits = 40

    commonXLims = (0, R.p['duration'] / second)
    voltageT = R.selectTrialT[1:]

    voltageE = R.selectTrialVExc[useTrialInd, :] - decreaseVoltageBy
    vSpikeTimesE = R.selectTrialSpikeExcT[useTrialInd][R.selectTrialSpikeExcI[useTrialInd] == useUnitInd]
    vSpikeTimesIndE = (vSpikeTimesE / R.p['dt'] * second / frameMult).astype(int)
    voltageE[vSpikeTimesIndE] = 0

    voltageI = R.selectTrialVInh[useTrialInd, :] - decreaseVoltageBy
    vSpikeTimesI = R.selectTrialSpikeInhT[useTrialInd][R.selectTrialSpikeInhI[useTrialInd] == useUnitInd]
    vSpikeTimesIndI = (vSpikeTimesI / R.p['dt'] * second / frameMult).astype(int)
    voltageI[vSpikeTimesIndI] = 0

    FRT = np.arange(FRDT / 2, R.p['duration'] / second - FRDT / 2, FRDT)
    FRE = R.selectTrialFRExc[useTrialInd, :]
    FRI = R.selectTrialFRInh[useTrialInd, :]

    spikeMonExcT = R.selectTrialSpikeExcT[useTrialInd]
    spikeMonExcI = R.selectTrialSpikeExcI[useTrialInd]
    spikeMonInhT = R.selectTrialSpikeInhT[useTrialInd]
    spikeMonInhI = R.selectTrialSpikeInhI[useTrialInd]

    downSampleE = np.random.choice(R.p['nExc'], size=targetDisplayedExcUnits, replace=False)
    downSampleI = np.random.choice(R.p['nInh'], size=targetDisplayedInhUnits, replace=False)
    matchingEUnitsBool = np.isin(spikeMonExcI, downSampleE)
    matchingIUnitsBool = np.isin(spikeMonInhI, downSampleI)
    DownSampleERev = np.full((downSampleE.max() + 1,), np.nan)
    DownSampleERev[downSampleE] = np.arange(downSampleE.size)
    DownSampleIRev = np.full((downSampleI.max() + 1,), np.nan)
    DownSampleIRev[downSampleI] = np.arange(downSampleI.size)
    spikesExcT = spikeMonExcT[matchingEUnitsBool]
    spikesExcI = DownSampleERev[spikeMonExcI[matchingEUnitsBool].astype(int)]
    spikesInhT = spikeMonInhT[matchingIUnitsBool]
    spikesInhI = targetDisplayedExcUnits + DownSampleIRev[spikeMonInhI[matchingIUnitsBool].astype(int)]

    fig1, ax1 = plt.subplots(4, 1, num=2, figsize=(5, 9), gridspec_kw={'height_ratios': [2, 1, 1, 1]}, sharex=True)

    srAx, frAx, veAx, viAx = ax1

    srAx.set(xlim=commonXLims, ylim=(0, 200), ylabel='Unit Index')      # spike raster
    frAx.set(xlim=commonXLims, ylim=(0, 30), ylabel='Firing Rate (Hz)')       # FR
    veAx.set(xlim=commonXLims, ylim=voltageYLims, ylabel='mV')  # voltageE
    viAx.set(xlim=commonXLims, ylim=voltageYLims, ylabel='mV', xlabel='Time (s)')  # voltageI

    srHandleE, = srAx.plot([], [], c='cyan', ls='', marker='.', markersize=1)
    srHandleI, = srAx.plot([], [], c='r', ls='', marker='.', markersize=1)

    frHandleE, = frAx.plot([], [], c='cyan', label='Exc', alpha=0.5)
    frHandleI, = frAx.plot([], [], c='r', label='Inh', alpha=0.5)

    vHandleE, = veAx.plot([], [], c='cyan', lw=.3)
    vHandleI, = viAx.plot([], [], c='r', lw=.3)

    vHandleE.set_data(voltageT, voltageE)
    vHandleI.set_data(voltageT, voltageI)

    frHandleE.set_data(FRT, FRE)
    frHandleI.set_data(FRT, FRI)
    srHandleE.set_data(spikesExcT, spikesExcI)
    srHandleI.set_data(spikesInhT, spikesInhI)

    fig1.suptitle('Trial {}'.format(actualTrialIndex))

    return fig1, ax1


def calculate_net_current_units(R):
    wEEMat = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nExc'], R.preEE, R.posEE, R.wEE_final)
    wEIMat = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nExc'], R.preEI, R.posEI, R.wEI_final)
    wIEMat = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nInh'], R.preIE, R.posIE, R.wIE_final)
    wIIMat = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nInh'], R.preII, R.posII, R.wII_final)
    totExcOntoExc = np.nansum(wEEMat, 0)
    totInhOntoExc = np.nansum(wEIMat, 0)
    totExcOntoInh = np.nansum(wIEMat, 0)
    totInhOntoInh = np.nansum(wIIMat, 0)
    netCurrentExc = totExcOntoExc - totInhOntoExc
    netCurrentInh = totExcOntoInh - totInhOntoInh
    return netCurrentExc, netCurrentInh
