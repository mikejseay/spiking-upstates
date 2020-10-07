from brian2 import *
from params import paramsDestexhe as p
from network import DestexheNetwork

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['simName'] = 'classicDestexhe'
p['saveWithDate'] = True

p['duration'] = 10 * second
p['nUnits'] = 1e4
p['propConnect'] = 0.05  # recurrent connection probability
p['bExc'] = 40 * pA  # adaptation param, decrease to get longer Up states

p['nPoissonInputUnits'] = p['nUnits']  # external units providing uncorrelated poisson inputs to recurrent
p['poissonInputsCorrelated'] = False  # whether different recurrent units receive correlated inputs or not
p['poissonInputRate'] = 0.315 * Hz

# p['poissonInputsCorrelated'] = True  # whether different recurrent units receive correlated inputs or not
# p['poissonInputRate'] = 0.315 * 0.6 * Hz

p['poissonDriveType'] = 'constant'  # ramp, constant, fullRamp
p['propConnectFeedforwardProjection'] = 0.05  # proportion of feedforward projections that are connected
p['poissonInputRateDivider'] = 1  # the factor by which to divide the rate
p['poissonInputWeightMultiplier'] = 1  # the factor by which to multiply the feedforward weight

# synaptic weights
p['qExc'] = 0.6 * uS  # will be divided by total # exc units and proportion of recurrent connectivity
p['qInh'] = 0.5 * uS  # will be divided by total # inh units and proportion of recurrent connectivity

# will be divided by total # input units & proportion of feedfoward projection,
# and multiplied by the poissonInputWeightMultiplier
p['qExcFeedforward'] = 0.6 * uS

if p['poissonInputWeightMultiplier'] is not 1:
    p['poissonInputRate'] /= (p['poissonInputWeightMultiplier'] ** 2)
    p['qExcFeedforward'] *= p['poissonInputWeightMultiplier']


# EXPERIMENTAL PARAMS (INTEGRATE LATER)
APPLY_WEAK_INPUTS = False
APPLY_STRONG_INPUTS = True
nCorrelatedInputUnits = 40
correlatedInputConnectionType = 'projection'  # projection or one-to-one
propConnectCorrelatedInput = 0.05
correlatedInputRate = 0.15 * Hz
nStrongStimEvents = 2
spacing = 10 * second
useQExcFeedforwardStrongExc = 15 * nS  # 18
useQExcFeedforwardStrongInh = 15 * nS  # 18.5

# start-ish
defaultclock.dt = p['dt']

# must pass in what's needed here...

N = DestexheNetwork(p)
N.build()
N.run()
N.save_results()
N.save_params()
