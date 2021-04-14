from brian2 import set_device, defaultclock, ms, second, pA, nA, amp, Hz
from params import paramsJercog as p
from params import paramsJercogEphysBuono
from network import JercogNetwork, JercogEphysNetwork
from results import Results, ResultsEphys
import numpy as np
import matplotlib.pyplot as plt
from generate import convert_kicks_to_current_series

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
ephysParams = paramsJercogEphysBuono.copy()
protectedKeys = ('nUnits', 'propInh', 'duration')
for pK in protectedKeys:
    del ephysParams[pK]

if USE_NEW_EPHYS_PARAMS:
    p.update(ephysParams)

# define parameters
setUpFRExc = 5 * Hz
setUpFRInh = 14 * Hz
tauUpFRTrials = 10
useRule = 'cross-homeo'  # cross-homeo or balance
kickType = 'kick'

if useRule == 'cross-homeo':
    alpha1 = 0.01 * pA / Hz  #
    alpha2 = None
    tauPlasticityTrials = None
    alphaPlasticity = None
elif useRule == 'balance':
    alpha1 = 0.05 * pA * pA / Hz / Hz / Hz
    alpha2 = 0.0005 * pA * pA / Hz / Hz / Hz
    tauPlasticityTrials = 100
    alphaPlasticity = 1 / tauPlasticityTrials * Hz
else:
    alpha1 = None
    alpha2 = None
    tauPlasticityTrials = None
    alphaPlasticity = None

# not used yet...
minAllowedWEE = 0.1
minAllowedWEI = 0.1
minAllowedWIE = 0.1
minAllowedWII = 0.1

nTrials = 2
# saveTrials = [1, int(0.4 * nTrials), nTrials]
saveTrials = [1, 3, 5, 8, 13, 20]  # 1-indexed
saveTrials = np.array(saveTrials) - 1

# simulation params
p['simName'] = 'jercogDefault'
p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])
indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)
kickTimes = [10 * ms, 1100 * ms]
kickSizes = [1]
p['duration'] = 1500 * ms
p['kickTimes'] = kickTimes
p['kickSizes'] = kickSizes
iKickRecorded = convert_kicks_to_current_series(p['kickDur'], p['kickTau'],
                                                p['kickTimes'], p['kickSizes'], p['duration'], p['dt'])

p['iKickRecorded'] = iKickRecorded


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

threshExc = RE.threshExc
threshInh = RE.threshInh
gainExc = RE.gainExc
gainInh = RE.gainInh

# set up network, experiment, and start recording
JN = JercogNetwork(p)
JN.initialize_network()
JN.initialize_recurrent_synapses2()

if kickType == 'kick':
    JN.initialize_units_kickable()
    JN.prepare_upCrit_experiment(minUnits=200, maxUnits=200, unitSpacing=5, timeSpacing=1500 * ms,
                             startTime=10 * ms, critExc=p['critExc'], critInh=p['critInh'])
else:
    JN.initialize_units()
    JN.set_kicked_units(onlyKickExc=p['onlyKickExc'])

# 185
JN.create_monitors()

# initialize history variables
trialUpFRExc = np.empty((nTrials,))
trialUpFRInh = np.empty((nTrials,))
trialwEE = np.empty((nTrials,))
trialwEI = np.empty((nTrials,))
trialwIE = np.empty((nTrials,))
trialwII = np.empty((nTrials,))

# initalize variables to represent the rolling average firing rates of the Exc and Inh units
# we start at None because this is undefined, and we will initialize at the exact value of the first UpFR
movAvgUpFRExc = None
movAvgUpFRInh = None

# get the initial weights (in amps here)
# wEE_init = JN.unitsExc.jE[0]
# wEI_init = JN.unitsExc.jI[0]
# wIE_init = JN.unitsInh.jE[0]
# wII_init = JN.unitsInh.jI[0]

wEE_init = JN.unitsExc.jE[0] / 2
wEI_init = JN.unitsExc.jI[0] / 2
wIE_init = JN.unitsInh.jE[0] / 2
wII_init = JN.unitsInh.jI[0] / 2

wEE = wEE_init
wEI = wEI_init
wIE = wIE_init
wII = wII_init

# store the network state
JN.N.store()

for trialInd in range(nTrials):

    # restore the initial network state
    JN.N.restore()

    # set the weights
    JN.unitsExc.jE = wEE
    JN.unitsExc.jI = wEI
    JN.unitsInh.jE = wIE
    JN.unitsInh.jI = wII

    # run the simulation
    JN.run()

    # calculate and record the average FR in the up state
    R = Results()
    R.init_from_network_object(JN)
    R.calculate_spike_rate()
    R.calculate_upstates()
    R.calculate_FR_in_upstates_simply()

    # save numerical results and/or plots!!!
    if trialInd in saveTrials:
        R.calculate_voltage_histogram(removeMode=True)
        R.reshape_upstates()
        R.calculate_FR_in_upstates()

        fig1, ax1 = plt.subplots(4, 1, num=1, figsize=(5, 9), gridspec_kw={'height_ratios': [3, 2, 1, 1]},
                                 sharex=True)
        R.plot_spike_raster(ax1[0])
        R.plot_firing_rate(ax1[1])
        ax1[1].set_ylim(0, 40)
        R.plot_voltage_detail(ax1[2], unitType='Exc', useStateInd=0)
        R.plot_updur_lines(ax1[2])
        R.plot_voltage_detail(ax1[3], unitType='Inh', useStateInd=0)
        R.plot_updur_lines(ax1[3])
        ax1[3].set(xlabel='Time (s)')
        R.plot_voltage_histogram_sideways(ax1[2], 'Exc')
        R.plot_voltage_histogram_sideways(ax1[3], 'Inh')
        fig1.suptitle(R.p['simName'] + '_' + useRule + '_t' + str(trialInd))
        fig1.savefig(R.p['saveFolder'] + '/' + R.rID + '_' + useRule + '_t' + str(trialInd) + '.png')
        fig1.clf()

    # if there was not zero Up states
    if len(R.ups) == 1:
        trialUpFRExc[trialInd] = R.upstateFRExc.mean()  # in Hz
        trialUpFRInh[trialInd] = R.upstateFRInh.mean()  # in Hz
        print('there was exactly one Up of duration {:.2f} s'.format(R.upDurs[0]))
    elif len(R.ups) > 1:
        print('for some reason there were multiple up states!!!')
        break
    else:
        trialUpFRExc[trialInd] = 0
        trialUpFRInh[trialInd] = 0

    # record the current weights in pA (divide to remove unit)
    trialwEE[trialInd] = wEE / pA
    trialwEI[trialInd] = wEI / pA
    trialwIE[trialInd] = wIE / pA
    trialwII[trialInd] = wII / pA

    # heck it, print those values
    print(
        'upstateFRExc: {:.2f} Hz, upstateFRInh: {:.2f} Hz, wEE: {:.2f} pA, wEI: {:.2f} pA, wIE: {:.2f} pA, wII: {:.2f} pA'.format(
            trialUpFRExc[trialInd], trialUpFRInh[trialInd], trialwEE[trialInd], trialwEI[trialInd], trialwIE[trialInd],
            trialwII[trialInd]))

    # calculate the moving average of the up FRs
    if movAvgUpFRExc:
        # movAvgUpFRExc += (-movAvgUpFRExc + trialUpFRExc[trialInd]) / tauUpFRTrials
        # movAvgUpFRInh += (-movAvgUpFRInh + trialUpFRInh[trialInd]) / tauUpFRTrials
        movAvgUpFRExc = trialUpFRExc[trialInd] * Hz
        movAvgUpFRInh = trialUpFRInh[trialInd] * Hz
    else:  # this only gets run the first trial (when they are None)
        movAvgUpFRExc = trialUpFRExc[trialInd] * Hz
        movAvgUpFRInh = trialUpFRInh[trialInd] * Hz

    # calculate the new weights, i.e. apply plasticity (cross homeo for now)
    # NOTE: we should be changing the weights by about 1% or less -- tune alpha and units to do so
    if useRule == 'cross-homeo':
        dwEE = alpha1 * (setUpFRInh - movAvgUpFRInh)
        dwEI = -alpha1 * (setUpFRInh - movAvgUpFRInh)
        dwIE = -alpha1 * (setUpFRExc - movAvgUpFRExc)
        dwII = alpha1 * (setUpFRExc - movAvgUpFRExc)
        wEE += dwEE
        wEI += dwEI
        wIE += dwIE
        wII += dwII
    elif useRule == 'balance':
        dwEE = alpha1 * gainExc * movAvgUpFRExc * (setUpFRExc - movAvgUpFRExc)
        wEE += dwEE
        dwEI = alphaPlasticity * (
                    (setUpFRExc / setUpFRInh * wEE / Hz - setUpFRExc / setUpFRInh / gainExc - threshExc / setUpFRInh) - wEI /  Hz)
        wEI += dwEI
        dwIE = -alpha2 * gainInh * movAvgUpFRInh * (setUpFRInh - movAvgUpFRInh)
        wIE += dwIE
        dwII = alphaPlasticity * (
                ((wIE / Hz * setUpFRExc - threshInh) / setUpFRInh - 1 / gainInh) - wII / Hz)
        wII += dwII

    print(
        'movAvgUpFRExc: {:.2f} Hz, movAvgUpFRInh: {:.2f} Hz, dwEE: {:.2f} pA, dwEI: {:.2f} pA, dwIE: {:.2f} pA, dwII: {:.2f} pA'.format(
            movAvgUpFRExc, movAvgUpFRInh, dwEE / pA, dwEI / pA, dwIE / pA, dwII / pA))
