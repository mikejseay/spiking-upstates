"""
contains parameter dictionaries for different types of networks.
mainly intended to represent default values.
i.e. new analyses should copy the dictionary and alter values as needed.
"""

from brian2 import ms, second, mV, nA, pF, nS, volt, pA, Hz
import numpy as np
import pandas as pd

paramsJercog = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/',
    'simName': 'classicJercog',
    'paramSet': 'classicJercog',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 10 * second,
    'updateMethod': 'euler',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'sE', 'sI', 'uE', 'uI'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

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
    # 'kickAmplitudeExc': 220 * mV,
    # 'kickAmplitudeInh': 0.4 * 220 * mV,
    'kickAmplitudeExc': 2.2 * nA,
    'kickAmplitudeInh': 0.4 * 2.2 * nA,

    # unit params
    'noiseSigma': 2.5 * mV,
    'eLeakExc': 7.6 * mV,
    'eLeakInh': 6.5 * mV,
    'vResetExc': 14 * mV,
    'vResetInh': 14 * mV,
    'vThreshExc': 20 * mV,
    'vThreshInh': 20 * mV,
    'adaptTau': 500 * ms,
    'adaptStrengthExc': 15 * mV,
    'adaptStrengthInh': 0 * mV,
    'refractoryPeriodExc': 5 * ms,
    'refractoryPeriodInh': 2 * ms,
    'refractoryPeriod': 1 * ms,  # overridden by the above but necessary
    'membraneCapacitanceExc': 200 * pF,
    'membraneCapacitanceInh': 100 * pF,
    'gLeakExc': 10 * nS,
    'gLeakInh': 10 * nS,
    # 'betaAdaptExc': 3 * nA * ms,  # 'adaptStrengthExc': 15 * mV, ... no idea!
    'betaAdaptExc': 3 * nA * ms,  # 'adaptStrengthExc': 15 * mV, ... no idea!
    'betaAdaptInh': 0 * nA * ms,  # 'adaptStrengthInh': 0 * mV,
    # i think it's supposed to be 15 mV * 10 nS * 20 (membrane tau) = 3 nA...

    # synaptic params
    # 'jEE': 280 * mV,
    # 'jEI': 70 * mV,
    # 'jIE': 500 * mV,
    # 'jII': 100 * mV,
    'jEE': 2.8 * nA,  # 280 mV * 10 nS
    'jEI': 0.7 * nA,  # 70 mV * 10 nS
    'jIE': 5 * nA,  # 500 mV * 10 nS
    'jII': 1 * nA,  # 100 mV * 10 nS
    'tauRiseExc': 8 * ms,
    'tauFallExc': 23 * ms,
    'tauRiseInh': 1 * ms,
    'tauFallInh': 1 * ms,
    'delayExc': 1 * ms,
    'delayInh': 0.5 * ms,
    'scaleWeightsByPConn': True,
    'critExc': 0.784 * volt,
    'critInh': 0.67625 * volt,

}

paramsJercogEphysBuono = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'simName': 'classicJercogEphysBuonoBen1',
    'paramSet': 'buonoEphysBen1',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 250 * ms,
    'updateMethod': 'euler',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 2,
    'propInh': 0.5,

    # unit params
    'eLeakExc': -65 * mV,
    'eLeakExc2': -65 * mV,
    'vResetExc': -58 * mV,
    'vResetExc2': -58 * mV,
    'vThreshExc': -52 * mV,
    'vThreshExc2': -46 * mV,
    'adaptTau': 500 * ms,
    'betaAdaptExc': 12 * nA * ms,
    'betaAdaptExc2': 12 * nA * ms,
    'refractoryPeriodExc': 2.5 * ms,
    'refractoryPeriodExc2': 2.5 * ms,
    'refractoryPeriod': 1 * ms,  # overridden by the above but necessary
    'membraneCapacitanceExc': 200 * pF,  # dictated by surface area
    'membraneCapacitanceExc2': 240 * pF,  # so it's smaller for inhibitory neurons
    'gLeakExc': 10 * nS,  # in theory dictated by density of Na+/K+ pump, etc
    'gLeakExc2': 8 * nS,  # so it's the same between the two

    # ephys params

    'noiseSigma': 1 * mV,
    'iExtRange': np.linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    'eLeakInh': -65 * mV,
    'vResetInh': -58 * mV,
    'vThreshInh': -43 * mV,
    'betaAdaptInh': 1 * nA * ms,
    'refractoryPeriodInh': 1 * ms,
    'membraneCapacitanceInh': 120 * pF,  # so it's smaller for inhibitory neurons
    'gLeakInh': 8 * nS,  # so it's the same between the two

    'jEE': 2.8 * nA,
    'jEI': 0.7 * nA,
    'jIE': 5 * nA,
    'jII': 1 * nA,
    'tauRiseExc': 8 * ms,
    'tauFallExc': 23 * ms,
    'tauRiseInh': 1 * ms,
    'tauFallInh': 1 * ms,
    'delayExc': 1 * ms,
    'delayInh': 0.5 * ms,
    'scaleWeightsByPConn': True,

    # additional params
    'useRule': 'upPoisson',
    'useOldWeightMagnitude': True,
    'disableWeightScaling': True,
    'applyLogToFR': False,
    'setMinimumBasedOnBalance': False,
    'recordMovieVariables': False,
}

jP = (
    # sim
    ('dt', 'Time step (ms)', ms),

    # net
    ('nUnits', 'Total # of units', 1),
    ('propInh', 'Proportion inhibitory units', 1),
    ('propConnect', 'Probability of connection', 1),

    # units
    ('eLeakExc', 'E Resting potential (mV)', mV),
    ('eLeakInh', 'I Resting potential (mV)', mV),
    ('vResetExc', 'E Reset potential (mV)', mV),
    ('vResetInh', 'I Reset potential (mV)', mV),
    ('vThreshExc', 'E Spike threshold (mV)', mV),
    ('vThreshInh', 'I Spike threshold (mV)', mV),
    ('refractoryPeriodExc', 'E Refractory period (ms)', ms),
    ('refractoryPeriodInh', 'I Refractory period (ms)', ms),
    ('membraneCapacitanceExc', 'E Membrane capacitance (pF)', pF),
    ('membraneCapacitanceInh', 'E Membrane capacitance (pF)', pF),
    ('gLeakExc', 'E Leak conductance (nS)', nS),
    ('gLeakInh', 'I Leak conductance (nS)', nS),
    ('betaAdaptExc', 'E Adaptation strength (nA * ms)', nA * ms),
    ('betaAdaptInh', 'I Adaptation strength (nA * ms)', nA * ms),
    ('adaptTau', 'Adaptation time constant (ms)', ms),
    ('noiseSigma', 'Noise amplitude (mV)', mV),

    # synapses
    ('jEE', 'Total E-to-E weight (pA)', pA),
    ('jIE', 'Total E-to-I weight (pA)', pA),
    ('jEI', 'Total I-to-E weight (pA)', pA),
    ('jII', 'Total I-to-I weight (pA)', pA),
    ('tauRiseExc', 'Excitatory rise time (ms)', ms),
    ('tauFallExc', 'Excitatory fall time (ms)', ms),
    ('tauRiseInh', 'Inhibitory rise time (ms)', ms),
    ('tauFallInh', 'Inhibitory fall time (ms)', ms),
    ('delayExc', 'Mean excitatory synaptic delay (ms)', ms),
    ('delayInh', 'Mean inhibitory synaptic delay (ms)', ms),

    # training / experiment
    ('setUpFRExc', 'Excitatory unit set-point (Hz)', Hz),
    ('setUpFRInh', 'Inhibitory unit set-point (Hz)', Hz),
    ('tauUpFRTrials', 'Upstate firing rate moving average time constant (trials)', 1),
    ('useRule', 'Learning rule', 's'),
    ('kickType', 'Kick method', 's'),
    ('initWeightMethod', 'Weight initialization method', 's'),
    ('maxAllowedFRExc', 'Max. allowed E FR for learning (Hz)', 1),  # should be Hz but 1
    ('maxAllowedFRInh', 'Max. allowed I FR for learning (Hz)', 1),  # should be Hz but 1
    ('nTrials', '# of learning trials', 's'),
    ('nUnitsToSpike', '# of units targeted by kick', 1),
    ('spikeInputAmplitude', 'Kick amplitude (pA)', pA),
    ('alpha1', 'Alpha 1 (pA)', pA),
    ('alpha2', 'Alpha 2 (pA)', pA),
    ('tauPlasticityTrials', '# of trials to balance', 1),
    ('alphaBalance', 'Alpha Balance', 1),
    ('minAllowedWEE', 'Min. allowed E-to-E weight (pA)', pA),
    ('minAllowedWEI', 'Min. allowed E-to-I weight (pA)', pA),
    ('minAllowedWIE', 'Min. allowed I-to-E weight (pA)', pA),
    ('minAllowedWII', 'Min. allowed I-to-I weight (pA)', pA),
    ('onlyKickExc', 'Whether to kick only E units', 1),
    ('threshExc', 'E unit threshold (pA)', pA),
    ('threshInh', 'I unit threshold (pA)', pA),
    ('gainExc', 'E unit gain (Hz / pA)', Hz / pA),
    ('gainInh', 'I unit gain (Hz / pA)', Hz / pA),
    ('wEEScale', 'E-to-E weight change scaling factor', 1),
    ('wIEScale', 'E-to-I weight change scaling factor', 1),
    ('wEIScale', 'I-to-E weight change scaling factor', 1),
    ('wIIScale', 'I-to-I weight change scaling factor', 1),
)


def create_param_df(p, paramNamer, removeParenthetical=True):
    cellTextList = []
    for shortName, longName, unit in paramNamer:
        if removeParenthetical and longName[-1] == ')':
            displayLongName, unitCrap = longName.split('(')
        else:
            displayLongName = longName
            unitCrap = ''
        try:
            if unit == 's':
                value = p[shortName]
                displayUnit = ''
            else:
                value = p[shortName] / unit
                displayUnit = unitCrap[:-1]
        except KeyError:
            print(shortName, 'was not in the param dict that was given')
            continue
        try:
            if value.is_integer():
                displayValue = '{:.0f}'.format(value)
            else:
                displayValue = value
        except AttributeError:
            displayValue = value
        if len(str(displayValue)) > 15:
            displayValue = str(int(float(displayValue)))
        cellTextList.append([displayLongName, displayValue, displayUnit])
    df = pd.DataFrame(cellTextList, columns=['Parameter', 'Value', 'Unit'])
    return df
