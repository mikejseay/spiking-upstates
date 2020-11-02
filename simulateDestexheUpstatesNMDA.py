from brian2 import *
from params import paramsDestexhe as p
from network import DestexheNetwork

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

p['duration'] = 5 * second
p['dt'] = 0.1 * ms

p['propConnect'] = 0.05  # recurrent connection probability
p['simName'] = 'classicDestexheNMDA0p05Conn'

p['bExc'] = 40 * pA  # adaptation param, decrease to get longer Up states 20-40ish
p['deltaVExc'] = 2 * mV   # 3
p['deltaVInh'] = 0.5 * mV
# p['vThresh'] = -45 * mV  # default is 50!!

APPLY_UNCORRELATED_INPUTS = False
APPLY_CORRELATED_INPUTS = True
CORRELATED_INPUTS_TARGET_EXC = True

# uncorrelated Poisson inputs (one-to-one, rate is usually multiplied by # of feedforward synapses per unit)
p['propConnectFeedforwardProjectionUncorr'] = 0.05  # proportion of feedforward projections that are connected
p['nPoissonUncorrInputUnits'] = p['nUnits']
p['nUncorrFeedforwardSynapsesPerUnit'] = int(p['propConnectFeedforwardProjectionUncorr'] *
                                        p['nPoissonUncorrInputUnits'] * (1 - p['propInh']))
p['poissonUncorrInputRate'] = 0.315 * p['nUncorrFeedforwardSynapsesPerUnit'] * Hz
p['qExcFeedforwardUncorr'] = 0.6 / p['nUncorrFeedforwardSynapsesPerUnit'] * uS
# p['qExcFeedforwardUncorr'] = 0.65 / p['nUncorrFeedforwardSynapsesPerUnit'] * uS

# correlated Poisson inputs (feedforward projection with shared targets)
p['propConnectFeedforwardProjectionCorr'] = 0.05  # proportion of feedforward projections that are connected
p['poissonCorrInputRate'] = 0.15 * Hz
p['nPoissonCorrInputUnits'] = 40
p['qExcFeedforwardCorr'] = 10 * nS

p['poissonDriveType'] = 'constant'  # ramp, constant, fullRamp
p['poissonInputRateDivider'] = 1  # the factor by which to divide the rate
p['poissonInputWeightMultiplier'] = 1  # the factor by which to multiply the feedforward weight

# synaptic weights
# normally 0.6
p['qExc'] = 0.3 * uS  # will be divided by total # exc units and proportion of recurrent connectivity
p['qInh'] = 0.5 * uS  # will be divided by total # inh units and proportion of recurrent connectivity

p['nInh'] = int(p['propInh'] * p['nUnits'])
p['nExc'] = int(p['nUnits'] - p['nInh'])
nRecurrentExcitatorySynapsesPerUnit = int(p['nExc'] * p['propConnect'])
useQExc = p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
useQNMDA = 0.1 * useQExc
nRecurrentInhibitorySynapsesPerUnit = int(p['nInh'] * p['propConnect'])
useQInh = p['qInh'] / nRecurrentInhibitorySynapsesPerUnit

nSamples = int(p['duration'] / defaultclock.dt)
iExtArray = np.zeros((nSamples,))
iExtRecorded = TimedArray(iExtArray, dt=defaultclock.dt)

if p['poissonInputWeightMultiplier'] is not 1:
    p['poissonInputRate'] /= (p['poissonInputWeightMultiplier'] ** 2)
    p['qExcFeedforward'] *= p['poissonInputWeightMultiplier']

# start-ish
defaultclock.dt = p['dt']

# must pass in what's needed here...

DN = DestexheNetwork(p)
DN.initialize_network()
DN.initialize_units_NMDA()

# DN.initialize_external_input()
# DN.initialize_external_input_memory(p['qExcFeedforward'], p['poissonInputRate'])
if APPLY_UNCORRELATED_INPUTS:
    DN.initialize_external_input_uncorrelated()
if APPLY_CORRELATED_INPUTS:
    DN.initialize_external_input_correlated(targetExc=CORRELATED_INPUTS_TARGET_EXC)

DN.initialize_recurrent_excitation_NMDA(useQExc=useQExc)
DN.create_monitors()
DN.run_NMDA(iExtRecorded=iExtRecorded, useQNMDA=useQNMDA)
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
