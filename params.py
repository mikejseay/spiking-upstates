from brian2 import *

paramsJercog = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/',
    'simName': 'classicJercog',
    'saveWithDate': True,
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

paramsJercog['nIncInh'] = int(paramsJercog['propInh'] * paramsJercog['nUnits'])
paramsJercog['nIncExc'] = paramsJercog['nUnits'] - paramsJercog['nIncInh']

paramsDestexhe = {
    # save / figs?
    'saveFolder': 'C:/Users/mikejseay/Dropbox/UCLA/courses/covid-era-modeling/figs/',
    'simName': 'classicJercog',
    'saveWithDate': True,
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
    'recordStateVariables': ['v', 'w', 'ge', 'gi'],
    'nRecordStateExc': 1,
    'nRecordStateInh': 1,

    # network params
    'nUnits': 1e4,  # ***
    'propInh': 0.2,
    'propConnect': 0.05,  # ***
    #     'propKicked': 0.1,
    #     'propPoissonInputs': 0.05,
    'poissonInputRate': 0.315 * Hz,  # ***
    'poissonInputWeightMultiplier': 2,  # ***

    # external inputs
    'kickMode': 'fixed',  # fixed or Poisson
    'kickTau': 0.5 * ms,
    'kickDur': 2 * ms,
    'kickLambda': 0.4 * second,
    'kickTimes': [0.1, 0.8] * second,
    'kickSizes': [1, 1],
    'kickAmplitudeExc': 1.5 * nA,
    'kickAmplitudeInh': 1.5 * nA,

    # unit params
    'membraneCapacitance': 200 * pF,  # 150 or 200
    'gLeak': 10 * nS,
    'eLeakExc': -63 * mV,  # -65 or -63
    'eLeakInh': -65 * mV,
    'vThresh': -50 * mV,
    'deltaVExc': 2 * mV,
    'deltaVInh': 0.5 * mV,
    'aExc': 0 * nS,  # 0 or 4 * ns
    'aInh': 0 * nS,
    'bExc': 40 * pA,  # 20-60 pA
    'bInh': 0 * pA,
    'adaptTau': 500 * ms,
    'refractoryPeriod': 5 * ms,

    # synaptic params
    'qExtExc': 0.6 * uS,
    'qExc': 0.6 * uS,  # *** # 1.5 * nS * 1e4 * (1 - .2) * 0.05,
    'qInh': 0.5 * uS,  # *** # 5 * nS * 1e4 * .2 * 0.05,
    'tauSynExc': 5 * ms,
    'tauSynInh': 5 * ms,
    'eExcSyn': 0 * mV,
    'eInhSyn': -80 * mV,

}
