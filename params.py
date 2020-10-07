from brian2 import *

paramsJercog = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/',
    'saveFigs': True,
    'saveType': '.png',

    # global sim params
    'dt': 0.05 * ms,
    'duration': 10 * second,
    'updateMethod': 'euler',
    'reportType': 'stdout',
    'reportPeriod': 1 * second,
    'doProfile': True,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'sE', 'sI', 'uE', 'uI'],
    'nRecordStateExc': 1,
    'nRecordStateInh': 1,

    # network params
    'nUnits': 5000,
    'propInh': 0.2,
    'propConnect': 1,
    'propKicked': 0.1,

    # external inputs
    'kickMode': 'Poisson',  # fixed or Poisson
    'kickTau': 0.5 * ms,
    'kickDur': 2 * ms,
    'kickLambda': 0.4 * second,
    'kickMinimumISI': 1 * second,
    'kickMaximumISI': 10 * second,
    'kickTimes': [0.1, 0.8] * second,
    'kickSizes': [1, 1],
    'kickAmplitudeExc': 220 * mV,
    'kickAmplitudeInh': 0.4 * 220 * mV,

    # unit params
    'noiseSigma': 2.5 * mV,
    'vTauExc': 20 * ms,
    'vTauInh': 10 * ms,
    'eLeakExc': 7.6 * mV,
    'eLeakInh': 6.5 * mV,
    'vResetExc': 14 * mV,
    'vResetInh': 14 * mV,
    'vThreshExc': 20 * mV,
    'vThreshInh': 20 * mV,
    'adaptTau': 500 * ms,
    'adaptStrengthExc': 15 * mV,
    'adaptStrengthInh': 0 * mV,
    'refractoryPeriod': 0 * ms,

    # synaptic params
    'jEE': 280 * mV,
    'jEI': 70 * mV,
    'jIE': 500 * mV,
    'jII': 100 * mV,
    'tauRiseExc': 8 * ms,
    'tauFallExc': 23 * ms,
    'tauRiseInh': 1 * ms,
    'tauFallInh': 1 * ms,
    'delayExc': 1 * ms,
    'delayInh': 0.5 * ms,
    'scaleWeightsByPConn': True,

}

paramsJercog['nInh'] = int(paramsJercog['propInh'] * paramsJercog['nUnits'])
paramsJercog['nExc'] = paramsJercog['nUnits'] - paramsJercog['nInh']
paramsJercog['nExcSpikemon'] = int(paramsJercog['nExc'] * paramsJercog['propSpikemon'])
paramsJercog['nInhSpikemon'] = int(paramsJercog['nInh'] * paramsJercog['propSpikemon'])
