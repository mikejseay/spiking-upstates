from brian2 import set_device, defaultclock, ms, second, nS, nA, amp, Hz
from params import paramsDestexhe as p
from params import paramsDestexheEphysBuono, paramsDestexheEphysOrig
from network import DestexheNetwork, DestexheEphysNetwork
from results import Results, ResultsEphys
import numpy as np
import matplotlib.pyplot as plt
from generate import convert_kicks_to_current_series
from datetime import datetime
import dill
import pickle
import os
from matplotlib.backends.backend_pdf import PdfPages

# for using Brian2GENN
USE_BRIAN2GENN = False
if USE_BRIAN2GENN:
    import brian2genn

    set_device('genn', debug=False)

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

defaultclock.dt = p['dt']

# determine 'up crit' empirically

USE_NEW_EPHYS_PARAMS = False

# remove protected keys from the dict whose params are being imported
if USE_NEW_EPHYS_PARAMS:
    ephysParams = paramsDestexheEphysBuono.copy()
else:
    ephysParams = paramsDestexheEphysOrig.copy()

protectedKeys = ('nUnits', 'propInh', 'duration')
for pK in protectedKeys:
    del ephysParams[pK]

p.update(ephysParams)

# simulation params
p['simName'] = 'destexheDefaultP02'
p['propConnect'] = 0.2
p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])
indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)

# p['onlyKickExc'] = True
# p['propKicked'] = 0.1
# p['duration'] = 1500 * ms
# p['kickTimes'] = [100 * ms]
# p['kickSizes'] = [1]
# iKickRecorded = convert_kicks_to_current_series(p['kickDur'], p['kickTau'],
#                                                 p['kickTimes'], p['kickSizes'], p['duration'], p['dt'])
# p['iKickRecorded'] = iKickRecorded

# define parameters
p['setUpFRExc'] = 5 * Hz
p['setUpFRInh'] = 14 * Hz
p['tauUpFRTrials'] = 5
p['useRule'] = 'balance'  # cross-homeo or balance
p['kickType'] = 'spike'  # kick or spike

# first quickly run an EPhys experiment with the given params to calculate the thresh and gain
pForEphys = p.copy()
pForEphys['propInh'] = 0.5
pForEphys['duration'] = 250 * ms
pForEphys['iExtRange'] = np.linspace(0, .3, 31) * nA
DEN = DestexheEphysNetwork(pForEphys)
DEN.build_classic()
DEN.run()

RE = ResultsEphys()
RE.init_from_network_object(DEN)
RE.calculate_thresh_and_gain()

p['threshExc'] = RE.threshExc
p['threshInh'] = RE.threshInh
p['gainExc'] = RE.gainExc
p['gainInh'] = RE.gainInh

if p['useRule'] == 'cross-homeo':
    p['alpha1'] = 0.00001 * nS / Hz / p['propConnect']
    p['alpha2'] = None
    p['tauPlasticityTrials'] = None
    p['alphaPlasticity'] = None
    p['minAllowedWEE'] = 0.1 * nS / p['propConnect']
    p['minAllowedWEI'] = 0.1 * nS / p['propConnect']
    p['minAllowedWIE'] = 0.1 * nS / p['propConnect']
    p['minAllowedWII'] = 0.1 * nS / p['propConnect']
elif p['useRule'] == 'balance':
    p['alpha1'] = 0.001 * nS * nA / Hz / Hz / Hz / p['propConnect']
    p['alpha2'] = 0.00001 * nS * nA / Hz / Hz / Hz / p['propConnect']
    # p['alpha1'] = 0.001
    # p['alpha2'] = 0.00001
    p['tauPlasticityTrials'] = 1000
    p['alphaPlasticity'] = 1 / p['tauPlasticityTrials'] * Hz / p['propConnect']

    p['minAllowedWEE'] = 0.1 * nS / p['propConnect']
    p['minAllowedWEI'] = 0.1 * nS / p['propConnect']
    p['minAllowedWIE'] = 0.1 * nS / p['propConnect']
    p['minAllowedWII'] = 0.1 * nS / p['propConnect']

    # p['minAllowedWEI'] = 0.1 * nS / p['propConnect']
    # p['minAllowedWEE'] = p['setUpFRInh'] / p['setUpFRExc'] *\
    #                      p['minAllowedWEI'] + (1 / p['gainExc'] + p['threshExc'] / p['setUpFRExc']) * Hz
    # p['minAllowedWII'] = 0.1 * nS / p['propConnect']
    # p['minAllowedWIE'] = (p['minAllowedWII'] + 1 / p['gainInh'] * Hz) *\
    #                      p['setUpFRInh'] / p['setUpFRExc'] + p['threshInh'] / p['setUpFRExc'] * Hz

p['maxAllowedFRExc'] = 100
p['maxAllowedFRInh'] = 250

p['nTrials'] = 34
p['saveTrials'] = [1, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]  # 1-indexed
p['saveTrials'] = np.array(p['saveTrials']) - 1

# set up network, experiment, and start recording
DN = DestexheNetwork(p)
DN.initialize_network()

if p['kickType'] == 'kick':
    DN.initialize_units_kickable()
    DN.set_kicked_units(onlyKickExc=p['onlyKickExc'])
elif p['kickType'] == 'spike':
    DN.initialize_units_kickable()
    DN.prepare_upCrit_experiment(minUnits=110, maxUnits=110, unitSpacing=5, timeSpacing=1400 * ms, startTime=100 * ms)

DN.initialize_recurrent_synapses_4bundles_modifiable()
DN.create_monitors()

# initialize history variables
trialUpFRExc = np.empty((p['nTrials'],))
trialUpFRInh = np.empty((p['nTrials'],))
trialwEE = np.empty((p['nTrials'],))
trialwEI = np.empty((p['nTrials'],))
trialwIE = np.empty((p['nTrials'],))
trialwII = np.empty((p['nTrials'],))

# initalize variables to represent the rolling average firing rates of the Exc and Inh units
# we start at None because this is undefined, and we will initialize at the exact value of the first UpFR
movAvgUpFRExc = None
movAvgUpFRInh = None

# get the initial weights (in siemens here)
p['wEE_init'] = DN.synapsesEE.qEE[0]
p['wIE_init'] = DN.synapsesIE.qIE[0]
p['wEI_init'] = DN.synapsesEI.qEI[0]
p['wII_init'] = DN.synapsesII.qII[0]

# p['wEE_init'] =DN.synapsesEE.qEE[0] / 2
# p['wEI_init'] =DN.synapsesEI.qEI[0] / 2
# p['wIE_init'] =DN.synapsesIE.qIE[0] / 2
# p['wII_init'] =DN.synapsesII.qII[0] / 2

# p['wEE_init'] = np.random.rand() * ??
# p['wEI_init'] = np.random.rand() *
# p['wIE_init'] = np.random.rand() *
# p['wII_init'] = np.random.rand() *

wEE = p['wEE_init']
wEI = p['wEI_init']
wIE = p['wIE_init']
wII = p['wII_init']

# store the network state
DN.N.store()

# initialize the pdf
pdfObject = PdfPages(p['saveFolder'] + DN.saveName + '_' + p['useRule'] + '_trials.pdf')

figCounter = 1
for trialInd in range(p['nTrials']):

    print('starting trial {}'.format(trialInd + 1))

    # restore the initial network state
    DN.N.restore()

    # set the weights
    DN.synapsesEE.qEE = wEE
    DN.synapsesEI.qEI = wEI
    DN.synapsesIE.qIE = wIE
    DN.synapsesII.qII = wII

    # run the simulation
    t0 = datetime.now()
    DN.run()
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
    R.init_from_network_object(DN)
    R.calculate_spike_rate()
    R.calculate_upstates()
    R.calculate_FR_in_upstates_simply()

    # save numerical results and/or plots!!!
    if saveThisTrial:
        R.calculate_voltage_histogram(removeMode=True)
        R.reshape_upstates()

        fig1, ax1 = plt.subplots(5, 1, num=figCounter, figsize=(5, 9), gridspec_kw={'height_ratios': [3, 2, 1, 1, 1]},
                                 sharex=True)
        R.plot_spike_raster(ax1[0])
        R.plot_firing_rate(ax1[1])
        ax1[1].set_ylim(0, 40)
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
                        open(R.p['saveFolder'] + '/' + R.rID + '_' + p['useRule'] + '_t' + str(trialInd + 1) + '.pickle',
                             'wb'))

    # if there was not zero Up states
    if len(R.ups) == 1:
        trialUpFRExc[trialInd] = R.upstateFRExc.mean()  # in Hz
        trialUpFRInh[trialInd] = R.upstateFRInh.mean()  # in Hz
        print('there was exactly one Up of duration {:.2f} s'.format(R.upDurs[0]))
    elif len(R.ups) > 1:
        print('for some reason there were multiple up states!!!')
        break
    else:
        # if there were no Up states, just take the avg FR (near 0 in this case)
        trialUpFRExc[trialInd] = R.FRExc.mean()
        trialUpFRInh[trialInd] = R.FRInh.mean()

    # if the currently assessed upstateFR was higher than the saturated FRs of the two types, reduce it
    if trialUpFRExc[trialInd] > p['maxAllowedFRExc']:
        trialUpFRExc[trialInd] = p['maxAllowedFRExc']
    if trialUpFRInh[trialInd] > p['maxAllowedFRInh']:
        trialUpFRInh[trialInd] = p['maxAllowedFRInh']

    # record the current weights in nS (divide to remove unit)
    trialwEE[trialInd] = wEE / nS
    trialwEI[trialInd] = wEI / nS
    trialwIE[trialInd] = wIE / nS
    trialwII[trialInd] = wII / nS

    # heck it, print those values
    print(
        'upstateFRExc: {:.2f} Hz, upstateFRInh: {:.2f} Hz, wEE: {:.2f} nS, wEI: {:.2f} nS, wIE: {:.2f} nS, wII: {:.2f} nS'.format(
            trialUpFRExc[trialInd], trialUpFRInh[trialInd], trialwEE[trialInd], trialwEI[trialInd], trialwIE[trialInd],
            trialwII[trialInd]))

    # calculate the moving average of the up FRs
    if movAvgUpFRExc:
        movAvgUpFRExc += (-movAvgUpFRExc + trialUpFRExc[trialInd] * Hz) / p['tauUpFRTrials']  # rolling average
        movAvgUpFRInh += (-movAvgUpFRInh + trialUpFRInh[trialInd] * Hz) / p['tauUpFRTrials']
        # movAvgUpFRExc = trialUpFRExc[trialInd] * Hz  # instantaneous
        # movAvgUpFRInh = trialUpFRInh[trialInd] * Hz
    else:  # this only gets run the first trial (when they are None)
        # movAvgUpFRExc = 0 * Hz  # initalize at zero
        # movAvgUpFRInh = 0 * Hz
        movAvgUpFRExc = trialUpFRExc[trialInd] * Hz  # initialize at the first measured
        movAvgUpFRInh = trialUpFRInh[trialInd] * Hz

    # calculate the new weights, i.e. apply plasticity (cross homeo for now)
    # NOTE: we should be changing the weights by about 1% or less -- tune alpha and units to do so
    if p['useRule'] == 'cross-homeo':
        dwEE = p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInh)
        dwEI = -p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInh)
        dwIE = -p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExc)
        dwII = p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExc)
        print(dwEE / wEE, dwEI / wEI, dwIE / wIE, dwII / wII)
        wEE += dwEE / DN.p['wEEScale']
        wEI += dwEI / DN.p['wEIScale']
        wIE += dwIE / DN.p['wIEScale']
        wII += dwII / DN.p['wIIScale']
    elif p['useRule'] == 'balance':
        dwEE = p['alpha1'] * p['gainExc'] * movAvgUpFRExc * (p['setUpFRExc'] - movAvgUpFRExc)
        wEE += dwEE / DN.p['wEEScale']
        # to make this dimensionally consistent, you can either measure the thresh / slope in siemens rather than amps
        # or you can multiply the siemens values by the average driving force you expect
        dwEI = p['alphaPlasticity'] * ((p['setUpFRExc'] / p['setUpFRInh'] * wEE / DN.p['wEEScale'] / Hz -
                                        p['setUpFRExc'] / p['setUpFRInh'] / p['gainExc'] -
                                        p['threshExc'] / p['setUpFRInh']) - wEI / DN.p['wEIScale'] / Hz)
        wEI += dwEI / DN.p['wEIScale']
        dwIE = -p['alpha2'] * p['gainInh'] * movAvgUpFRInh * (p['setUpFRInh'] - movAvgUpFRInh)
        wIE += dwIE / DN.p['wIEScale']
        dwII = p['alphaPlasticity'] * (((wIE / DN.p['wIEScale'] / Hz * p['setUpFRExc'] -
                                         p['threshInh']) / p['setUpFRInh'] -
                                        1 / p['gainInh']) - wII / DN.p['wIIScale'] / Hz)
        wII += dwII / DN.p['wIIScale']

    # our implementation of the min thing just undoes the weight change, lol
    if wEE < p['minAllowedWEE']:
        wEE += (p['minAllowedWEE'] - wEE)
        # wEE -= dwEE
    if wEI < p['minAllowedWEI']:
        wEI += (p['minAllowedWEI'] - wEI)
        # wEI -= dwEI
    if wIE < p['minAllowedWIE']:
        wIE += (p['minAllowedWIE'] - wIE)
        # wIE -= dwIE
    if wII < p['minAllowedWII']:
        wII += (p['minAllowedWII'] - wII)
        # wII -= dwII

    print(
        'movAvgUpFRExc: {:.2f} Hz, movAvgUpFRInh: {:.2f} Hz, dwEE: {:.2f} nS, dwEI: {:.2f} nS, dwIE: {:.2f} nS, dwII: {:.2f} nS'.format(
            movAvgUpFRExc, movAvgUpFRInh, dwEE / DN.p['wEEScale'] / nS, dwEI / DN.p['wEIScale'] / nS,
            dwIE / DN.p['wIEScale'] / nS, dwII / DN.p['wIIScale'] / nS))

# close pdf
pdfObject.close()

# params
savePath = os.path.join(p['saveFolder'], DN.saveName + '_params.pkl')
with open(savePath, 'wb') as f:
    dill.dump(p, f)

# results
savePath = os.path.join(DN.p['saveFolder'], DN.saveName + '_results.npz')
np.savez(savePath, trialUpFRExc=trialUpFRExc, trialUpFRInh=trialUpFRInh, trialwEE=trialwEE,
         trialwEI=trialwEI, trialwIE=trialwIE, trialwII=trialwII)
