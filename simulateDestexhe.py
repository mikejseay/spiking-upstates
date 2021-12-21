from brian2 import set_device, second, ms, mV, pA, uS, Hz, nS, defaultclock
from params import paramsDestexhe as p
from params import paramsDestexheEphysBuono, paramsDestexheEphysOrig
from network import DestexheNetwork
from generate import poisson_kicks_jercog
from results import Results
import matplotlib.pyplot as plt

# for using Brian2GENN
USE_BRIAN2GENN = False
if USE_BRIAN2GENN:
    from datetime import datetime
    t0 = datetime.now()
    import brian2genn
    set_device('genn', debug=False)

MAKE_UP_PLOTS = False
USE_NEW_EPHYS_PARAMS = True
KICKS_POISSON = True

# remove protected keys from the dict whose params are being imported
ephysParams = paramsDestexheEphysOrig.copy()
protectedKeys = ('nUnits', 'propInh', 'duration')
for pK in protectedKeys:
    del ephysParams[pK]

if USE_NEW_EPHYS_PARAMS:
    p.update(ephysParams)

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

p['nUnits'] = 1e4
p['propConnect'] = 0.1
p['duration'] = 10 * second
p['dt'] = 0.1 * ms

p['qExc'] = 0.6 * uS  # will be divided by total # exc units and proportion of recurrent connectivity
p['qInh'] = 0.5 * uS  # will be divided by total # inh units and proportion of recurrent connectivity

USE_DISTRIBUTED_WEIGHTS = False
normalMean = 1
normalSD = 0.2

APPLY_UNCORRELATED_INPUTS = True
APPLY_CORRELATED_INPUTS = False

USE_PRIOR_CORRELATED_INPUT_PATTERN = False  # note this will overwrite several paramters below
loadFolder = 'C:/Users/mikejseay/Documents/BrianResults/'
targetSim = 'destexheEphysBuono_2021-03-29-16-24'

MONITOR_CORRELATED_INPUTS = False
CORRELATED_INPUTS_TARGET_EXC = False
APPLY_KICKS = False

# uncorrelated Poisson inputs (one-to-one, rate is usually multiplied by # of feedforward synapses per unit)
p['propConnectFeedforwardProjectionUncorr'] = 0.05  # proportion of feedforward projections that are connected
p['nPoissonUncorrInputUnits'] = p['nUnits']

# p['nUncorrFeedforwardSynapsesPerUnit'] = int(p['propConnectFeedforwardProjectionUncorr'] *
#                                         p['nPoissonUncorrInputUnits'] * (1 - p['propInh']))
# p['poissonUncorrInputRate'] = 0.315 * p['nUncorrFeedforwardSynapsesPerUnit'] * Hz
# p['qExcFeedforwardUncorr'] = 0.6 * uS / p['nUncorrFeedforwardSynapsesPerUnit']

# to make 5e4 work...
# p['qExcFeedforwardUncorr'] = p['qExcFeedforwardUncorr'] * 1.5

# for the corr + uncorr inputs scenario...

p['nUncorrFeedforwardSynapsesPerUnit'] = int(p['propConnectFeedforwardProjectionUncorr'] *
                                        1e4 * (1 - p['propInh']))
p['poissonUncorrInputRate'] = 0.315 * p['nUncorrFeedforwardSynapsesPerUnit'] * Hz
p['qExcFeedforwardUncorr'] = 0.6 * uS / p['nUncorrFeedforwardSynapsesPerUnit']

# p['nUncorrFeedforwardSynapsesPerUnit'] = int(p['propConnectFeedforwardProjectionUncorr'] *
#                                         1e4 * (1 - p['propInh']))
# p['poissonUncorrInputRate'] = 0.16 * p['nUncorrFeedforwardSynapsesPerUnit'] * Hz
# p['qExcFeedforwardUncorr'] = 0.3 * uS / p['nUncorrFeedforwardSynapsesPerUnit']


# to try to make the 5e4 work with only uncorr inputs...
# p['poissonUncorrInputRate'] = 0.3168 * p['nUncorrFeedforwardSynapsesPerUnit'] * Hz
# p['qExcFeedforwardUncorr'] = 0.6138 * uS / p['nUncorrFeedforwardSynapsesPerUnit']

# correlated Poisson inputs (feedforward projection with shared targets)
p['propConnectFeedforwardProjectionCorr'] = 0.04  # proportion of feedforward projections that are connected

# IF USING INHERITED POISSON PATTERN THESE DON'T MATTER
p['poissonCorrInputRate'] = 0.2 * Hz
p['nPoissonCorrInputUnits'] = 1

p['qExcFeedforwardCorr'] = 18.5 * nS

p['poissonDriveType'] = 'constant'  # ramp, constant, fullRamp
p['poissonInputRateDivider'] = 1  # the factor by which to divide the rate
p['poissonInputWeightMultiplier'] = 1  # the factor by which to multiply the feedforward weight

# kicks
if APPLY_KICKS:
    p['propKicked'] = 0.02
    p['onlyKickExc'] = True
    kickTimes, kickSizes = poisson_kicks_jercog(p['kickLambda'], p['duration'],
                                                p['kickMinimumISI'], p['kickMaximumISI'])
    print(kickTimes)
    p['kickTimes'] = kickTimes
    p['kickSizes'] = kickSizes

# synaptic weights
# will be divided by total # input units & proportion of feedfoward projection,
# and multiplied by the poissonInputWeightMultiplier

# uncorrelated inputs
# nFeedforwardSynapsesPerUnit = int(p['propConnectFeedforwardProjection'] * p['nPoissonInputUnits'] *
#                                           (1 - p['propInh']))
# useQExcFeedforward = p['qExcFeedforward'] / nFeedforwardSynapsesPerUnit
# useExternalRate = p['poissonInputRate'] * nFeedforwardSynapsesPerUnit

if p['poissonInputWeightMultiplier'] is not 1:
    p['poissonInputRate'] /= (p['poissonInputWeightMultiplier'] ** 2)
    p['qExcFeedforward'] *= p['poissonInputWeightMultiplier']

indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)

# start-ish
defaultclock.dt = p['dt']

# must pass in what's needed here...

DN = DestexheNetwork(p)
DN.initialize_network()
DN.initialize_units()

# DN.initialize_external_input()
# DN.initialize_external_input_memory(p['qExcFeedforward'], p['poissonInputRate'])
if APPLY_UNCORRELATED_INPUTS:
    DN.initialize_external_input_uncorrelated()
if APPLY_CORRELATED_INPUTS:
    if USE_PRIOR_CORRELATED_INPUT_PATTERN:
        DN.initialize_prior_external_input_correlated(targetSim, loadFolder,
                                                      targetExc=CORRELATED_INPUTS_TARGET_EXC)
    else:
        DN.initialize_external_input_correlated(targetExc=CORRELATED_INPUTS_TARGET_EXC,
                                                monitorProcesses=MONITOR_CORRELATED_INPUTS)
if APPLY_KICKS:
    DN.set_spiked_units(onlySpikeExc=p['onlyKickExc'])

if USE_DISTRIBUTED_WEIGHTS:
    DN.initialize_recurrent_synapses_4bundles_distributed(normalMean=normalMean, normalSD=normalSD)
else:
    DN.initialize_recurrent_synapses_4bundles()

DN.create_monitors()
DN.run()
DN.save_results()
DN.save_params()

if USE_BRIAN2GENN:
    t1 = datetime.now()
    print('took:', t1 - t0)

R = Results(DN.saveName, DN.p['saveFolder'])
R.calculate_PSTH()
R.calculate_voltage_histogram(removeMode=True)
R.calculate_upstates()
if len(R.ups) > 0:
    R.reshape_upstates()
    R.calculate_FR_in_upstates()
    print('average FR in upstate for Exc: {:.2f}, Inh: {:.2f} '.format(R.upstateFRExc.mean(), R.upstateFRInh.mean()))

# quit()

fig1, ax1 = plt.subplots(2, 1, num=1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]},
                         sharex=True)
R.plot_spike_raster(ax1[0])
R.plot_firing_rate(ax1[1])

fig2, ax2 = plt.subplots(3, 1, num=2, figsize=(10, 9), sharex=True)
R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0)
R.plot_updur_lines(ax2[0])
R.plot_voltage_detail(ax2[1], unitType='Exc', useStateInd=1)
R.plot_updur_lines(ax2[1])
R.plot_voltage_detail(ax2[2], unitType='Inh', useStateInd=0)

fig2b, ax2b = plt.subplots(1, 1, num=21)
R.plot_updur_lines(ax2b)
R.plot_voltage_histogram(ax2b, yScaleLog=True)

if MAKE_UP_PLOTS:
    fig3, ax3 = plt.subplots(num=3, figsize=(10, 9))
    R.plot_state_duration_histogram(ax3)

    fig4, ax4 = plt.subplots(1, 2, num=4, figsize=(10, 9))
    R.plot_consecutive_state_correlation(ax4)

    fig5, ax5 = plt.subplots(2, 1, num=5, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]},
                             sharex=True)
    R.plot_upstate_voltage_image(ax5[0])
    R.plot_upstate_voltages(ax5[1])

    fig6, ax6 = plt.subplots(2, 1, num=6, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]},
                             sharex=True)
    R.plot_upstate_raster(ax6[0])
    R.plot_upstate_FR(ax6[1])
