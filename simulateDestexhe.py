from brian2 import *
from params import paramsDestexhe as p
from network import DestexheNetwork
from generate import generate_poisson_kicks_jercog

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

p['duration'] = 100 * second
p['dt'] = 0.1 * ms

# p['propConnect'] = 0.05  # recurrent connection probability
# p['simName'] = 'classicDestexhe0p05Conn'

# p['propConnect'] = 0.05  # recurrent connection probability
# p['simName'] = 'classicDestexhe0p05ConnSparseCorr'

# p['propConnect'] = 1  # recurrent connection probability
# p['simName'] = 'classicDestexheFullConn'

# p['propConnect'] = 0.1  # recurrent connection probability
# p['simName'] = 'classicDestexhe0p10ConnSparseCorr'

# p['propConnect'] = 0.2  # recurrent connection probability
# p['simName'] = 'classicDestexhe0p20ConnSparseCorr'

# p['propConnect'] = 0.5  # recurrent connection probability
# p['simName'] = 'c--lassicDestexhe0p5ConnSparseCorr'

# p['nUnits'] = 20e4
# p['propConnect'] = 0.05  # recurrent connection probability
# p['simName'] = 'classicDestexheFullConn200kUnits'

# p['nUnits'] = 1e3
# p['propConnect'] = 0.05  # recurrent connection probability
# p['simName'] = 'classicDestexhe0p05Conn1kUnits'

p['nUnits'] = 5e4
p['simName'] = 'destexhe0p05Conn5e4units'

p['bExc'] = 40 * pA  # adaptation param, decrease to get longer Up states 20-40ish
p['deltaVExc'] = 2 * mV   # 3
p['deltaVInh'] = 0.5 * mV
# p['vThresh'] = -45 * mV  # default is 50!!

p['qExc'] = 0.6 * uS  # will be divided by total # exc units and proportion of recurrent connectivity
p['qInh'] = 0.5 * uS  # will be divided by total # inh units and proportion of recurrent connectivity

APPLY_UNCORRELATED_INPUTS = True
APPLY_CORRELATED_INPUTS = False
CORRELATED_INPUTS_TARGET_EXC = True
APPLY_KICKS = True

# uncorrelated Poisson inputs (one-to-one, rate is usually multiplied by # of feedforward synapses per unit)
p['propConnectFeedforwardProjectionUncorr'] = 0.05  # proportion of feedforward projections that are connected
p['nPoissonUncorrInputUnits'] = p['nUnits']
# p['nUncorrFeedforwardSynapsesPerUnit'] = int(p['propConnectFeedforwardProjectionUncorr'] *
#                                         p['nPoissonUncorrInputUnits'] * (1 - p['propInh']))
# p['poissonUncorrInputRate'] = 0.315 * p['nUncorrFeedforwardSynapsesPerUnit'] * Hz
# p['qExcFeedforwardUncorr'] = 0.6 * uS / p['nUncorrFeedforwardSynapsesPerUnit']

p['nUncorrFeedforwardSynapsesPerUnit'] = int(p['propConnectFeedforwardProjectionUncorr'] *
                                        1e4 * (1 - p['propInh']))
# p['poissonUncorrInputRate'] = 0.315 * p['nUncorrFeedforwardSynapsesPerUnit'] * Hz
# p['qExcFeedforwardUncorr'] = 0.6 * uS / p['nUncorrFeedforwardSynapsesPerUnit']

p['poissonUncorrInputRate'] = 0.16 * p['nUncorrFeedforwardSynapsesPerUnit'] * Hz
p['qExcFeedforwardUncorr'] = 0.3 * uS / p['nUncorrFeedforwardSynapsesPerUnit']


# try halving the rate and amplitude of uncorrelated, while halving the amplitude of the correlated

# p['qExcFeedforwardUncorr'] = 0.6 * uS / 1e4

# correlated Poisson inputs (feedforward projection with shared targets)
p['propConnectFeedforwardProjectionCorr'] = 0.05  # proportion of feedforward projections that are connected
p['poissonCorrInputRate'] = 0.15 * Hz
p['nPoissonCorrInputUnits'] = 40
# p['qExcFeedforwardCorr'] = 15 * nS
p['qExcFeedforwardCorr'] = 12 * nS

p['poissonDriveType'] = 'constant'  # ramp, constant, fullRamp
p['poissonInputRateDivider'] = 1  # the factor by which to divide the rate
p['poissonInputWeightMultiplier'] = 1  # the factor by which to multiply the feedforward weight

# kicks
if APPLY_KICKS:
    p['propKicked'] = 0.02
    p['onlyKickExc'] = True
    kickTimes, kickSizes = generate_poisson_kicks_jercog(p['kickLambda'], p['duration'],
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
    DN.initialize_external_input_correlated(targetExc=CORRELATED_INPUTS_TARGET_EXC)
if APPLY_KICKS:
    DN.set_spiked_units(onlySpikeExc=p['onlyKickExc'])

DN.initialize_recurrent_synapses()
DN.create_monitors()
DN.run()
DN.save_results()
DN.save_params()

# EXPERIMENTAL PARAMS (INTEGRATE LATER)
# APPLY_WEAK_INPUTS = False
# APPLY_STRONG_INPUTS = True
# nCorrelatedInputUnits = 40
# correlatedInputConnectionType = 'projection'  # projection or one-to-one
# propConnectCorrelatedInput = 0.05
# correlatedInputRate = 0.15 * Hz
# nStrongStimEvents = 2
# spacing = 10 * second
# useQExcFeedforwardStrongExc = 15 * nS  # 18
# useQExcFeedforwardStrongInh = 15 * nS  # 18.5
