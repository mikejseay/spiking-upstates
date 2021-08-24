"""
contains parameter dictionaries for different types of networks.
mainly intended to represent default values.
i.e. new analyses should copy the dictionary and alter values as needed.
"""

from brian2 import *
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
    'refractoryPeriodExc': 2.5 * ms,
    'refractoryPeriodInh': 1 * ms,
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

paramsJercog['nIncInh'] = int(paramsJercog['propInh'] * paramsJercog['nUnits'])
paramsJercog['nIncExc'] = paramsJercog['nUnits'] - paramsJercog['nIncInh']

paramsJercogBen = {
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
    'adaptTau': 500 * ms,
    'refractoryPeriod': 1 * ms,  # overridden by the above but necessary

    'eLeakExc': 7.6 * mV,
    'vResetExc': 14 * mV,
    'vThreshExc': 20 * mV,
    'refractoryPeriodExc': 2.5 * ms,
    'membraneCapacitanceExc': 200 * pF,
    'gLeakExc': 10 * nS,
    'betaAdaptExc': 3 * nA * ms,  # 'adaptStrengthExc': 15 * mV, ... no idea!

    'eLeakExc2': 7.6 * mV,
    'vResetExc2': 14 * mV,
    'vThreshExc2': 24 * mV,
    'refractoryPeriodExc2': 2.5 * ms,
    'membraneCapacitanceExc2': 240 * pF,
    'gLeakExc2': 8 * nS,
    'betaAdaptExc2': 3 * nA * ms,  # 'adaptStrengthExc': 15 * mV, ... no idea!

    'eLeakInh': 6.5 * mV,
    'vResetInh': 14 * mV,
    'vThreshInh': 20 * mV,
    'refractoryPeriodInh': 1 * ms,
    'membraneCapacitanceInh': 100 * pF,
    'gLeakInh': 10 * nS,
    'betaAdaptInh': 0 * nA * ms,  # 'adaptStrengthInh': 0 * mV,
    # i think it's supposed to be 15 mV * 10 nS * 20 (membrane tau) = 3 nA...

    # synaptic params
    # 'jEE': 280 * mV,
    # 'jEI': 70 * mV,
    # 'jIE': 500 * mV,
    # 'jII': 100 * mV,

    'adaptStrengthExc': 15 * mV,  # not used anymore...
    'adaptStrengthInh': 0 * mV,

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

paramsJercogEphysOrig = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'simName': 'classicJercogEphysOrig',
    'paramSet': 'classicJercogEphys',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 250 * ms,
    # 'updateMethod': 'exact',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'eLeakExc': -58.9 * mV,  # 'eLeakExc': 7.6 * mV,
    'eLeakInh': -60 * mV,  # 'eLeakInh': 6.5 * mV,
    'vResetExc': -52.5 * mV,  # 'vResetExc': 14 * mV,
    'vResetInh': -52.5 * mV,  # 'vResetInh': 14 * mV,
    'vThreshExc': -46.5 * mV,  # 'vThreshExc': 20 * mV,
    'vThreshInh': -46.5 * mV,  # 'vThreshInh': 20 * mV,
    'adaptTau': 500 * ms,  # 'adaptTau': 500 * ms,
    'betaAdaptExc': 3 * nA * ms,  # 'adaptStrengthExc': 15 * mV, ... no idea! 3 or 4?
    'betaAdaptInh': 0 * nA * ms,  # 'adaptStrengthInh': 0 * mV,
    'refractoryPeriod': 0 * ms,  # 'refractoryPeriod': 0 * ms,
    'membraneCapacitanceExc': 200 * pF,
    'membraneCapacitanceInh': 100 * pF,
    'gLeakExc': 10 * nS,
    'gLeakInh': 10 * nS,

    # ephys params
    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    # synaptic params
    # 'jEE': 280 * mV,
    # 'jEI': 70 * mV,
    # 'jIE': 500 * mV,
    # 'jII': 100 * mV,
    'tauRiseExc': 8 * ms,
    'tauFallExc': 23 * ms,
    'tauRiseInh': 1 * ms,
    'tauFallInh': 1 * ms,
    'delayExc': 1 * ms,
    'delayInh': 0.5 * ms,
    'scaleWeightsByPConn': True,

}

paramsJercogEphysBuono = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'simName': 'classicJercogEphysBuono',
    'paramSet': 'buonoEphys',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 250 * ms,
    # 'updateMethod': 'exact',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'eLeakExc': -60 * mV,
    'eLeakInh': -60 * mV,
    'vResetExc': -60 * mV,
    'vResetInh': -60 * mV,
    'vThreshExc': -50 * mV,
    'vThreshInh': -40 * mV,
    'adaptTau': 500 * ms,
    'betaAdaptExc': 10 * nA * ms,
    'betaAdaptInh': 0 * nA * ms,
    'refractoryPeriodExc': 2.5 * ms,
    'refractoryPeriodInh': 1 * ms,
    'refractoryPeriod': 1 * ms,  # overridden by the above but necessary
    'membraneCapacitanceExc': 200 * pF,  # dictated by surface area
    'membraneCapacitanceInh': 100 * pF,  # so it's smaller for inhibitory neurons
    'gLeakExc': 10 * nS,  # in theory dictated by density of Na+/K+ pump, etc
    'gLeakInh': 10 * nS,  # so it's the same between the two

    # ephys params
    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    # synaptic params
    # 'jEE': 280 * mV,
    # 'jEI': 70 * mV,
    # 'jIE': 500 * mV,
    # 'jII': 100 * mV,
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

}

paramsJercogEphysBuono2 = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'simName': 'classicJercogEphysBuono',
    'paramSet': 'buonoEphys2',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 250 * ms,
    # 'updateMethod': 'exact',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'refractoryPeriod': 1 * ms,  # overridden by the above but necessary
    'adaptTau': 500 * ms,

    'eLeakExc2': -60 * mV,
    'vResetExc2': -60 * mV,
    'vThreshExc2': -52 * mV,  # -50 or -52
    'betaAdaptExc2': 24 * nA * ms,
    'refractoryPeriodExc2': 2.5 * ms,
    'membraneCapacitanceExc2': 200 * pF,  # dictated by surface area
    'gLeakExc2': 10 * nS,  # in theory dictated by density of Na+/K+ pump, etc

    'eLeakExc': -60 * mV,
    'vResetExc': -60 * mV,
    'vThreshExc': -52 * mV,  # -50 or -52
    'betaAdaptExc': 24 * nA * ms,
    'refractoryPeriodExc': 2.5 * ms,
    'membraneCapacitanceExc': 200 * pF,  # dictated by surface area
    'gLeakExc': 10 * nS,  # in theory dictated by density of Na+/K+ pump, etc

    'eLeakInh': -60 * mV,
    'vResetInh': -60 * mV,
    'vThreshInh': -42 * mV,
    'betaAdaptInh': 2 * nA * ms,
    'refractoryPeriodInh': 1 * ms,
    'membraneCapacitanceInh': 120 * pF,  # so it's smaller for inhibitory neurons
    'gLeakInh': 8 * nS,  # so it's the same between the two

    # ephys params

    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    # synaptic params
    # 'jEE': 280 * mV,
    # 'jEI': 70 * mV,
    # 'jIE': 500 * mV,
    # 'jII': 100 * mV,
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

}

paramsJercogEphysBuono22 = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'simName': 'classicJercogEphysBuonoBen1',
    'paramSet': 'buonoEphysBen1',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 250 * ms,
    # 'updateMethod': 'exact',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'eLeakExc': -60 * mV,
    'eLeakExc2': -60 * mV,
    'vResetExc': -60 * mV,
    'vResetExc2': -60 * mV,
    'vThreshExc': -52 * mV,
    'vThreshExc2': -44 * mV,
    'adaptTau': 500 * ms,
    'betaAdaptExc': 23 * nA * ms,
    'betaAdaptExc2': 23 * nA * ms,
    'refractoryPeriodExc': 2.5 * ms,
    'refractoryPeriodExc2': 2.5 * ms,
    'refractoryPeriod': 1 * ms,  # overridden by the above but necessary
    'membraneCapacitanceExc': 200 * pF,  # dictated by surface area
    'membraneCapacitanceExc2': 200 * pF,  # so it's smaller for inhibitory neurons
    'gLeakExc': 10 * nS,  # in theory dictated by density of Na+/K+ pump, etc
    'gLeakExc2': 7.5 * nS,  # so it's the same between the two

    # ephys params
    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    'eLeakInh': -60 * mV,
    'vResetInh': -60 * mV,
    'vThreshInh': -42 * mV,
    'betaAdaptInh': 2 * nA * ms,
    'refractoryPeriodInh': 1 * ms,
    'membraneCapacitanceInh': 120 * pF,  # so it's smaller for inhibitory neurons
    'gLeakInh': 8 * nS,  # so it's the same between the two

    # ephys params

    # synaptic params
    # 'jEE': 280 * mV,
    # 'jEI': 70 * mV,
    # 'jIE': 500 * mV,
    # 'jII': 100 * mV,
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

}

paramsJercogEphysBuono3 = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'simName': 'classicJercogEphysBuono',
    'paramSet': 'buonoEphys',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 250 * ms,
    # 'updateMethod': 'exact',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'eLeakExc': -60 * mV,
    'eLeakInh': -60 * mV,
    'vResetExc': -60 * mV,
    'vResetInh': -60 * mV,
    'vThreshExc': -50 * mV,
    'vThreshInh': -42.5 * mV,
    'adaptTau': 500 * ms,
    'betaAdaptExc': 10 * nA * ms,
    'betaAdaptInh': 2.5 * nA * ms,
    'refractoryPeriodExc': 2.5 * ms,
    'refractoryPeriodInh': 1 * ms,
    'refractoryPeriod': 1 * ms,  # overridden by the above but necessary
    'membraneCapacitanceExc': 200 * pF,  # dictated by surface area
    'membraneCapacitanceInh': 125 * pF,  # so it's smaller for inhibitory neurons
    'gLeakExc': 10 * nS,  # in theory dictated by density of Na+/K+ pump, etc
    'gLeakInh': 7.5 * nS,  # so it's the same between the two

    # ephys params
    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    # synaptic params
    # 'jEE': 280 * mV,
    # 'jEI': 70 * mV,
    # 'jIE': 500 * mV,
    # 'jII': 100 * mV,
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

}

paramsJercogEphysBuono4 = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'simName': 'classicJercogEphysBuonoBen1',
    'paramSet': 'buonoEphysBen1',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 250 * ms,
    # 'updateMethod': 'exact',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'eLeakExc': -60 * mV,
    'eLeakExc2': -60 * mV,
    'vResetExc': -60 * mV,
    'vResetExc2': -60 * mV,
    'vThreshExc': -47 * mV,
    'vThreshExc2': -43 * mV,
    'adaptTau': 500 * ms,
    'betaAdaptExc': 14.5 * nA * ms,
    'betaAdaptExc2': 14.5 * nA * ms,
    'refractoryPeriodExc': 2.5 * ms,
    'refractoryPeriodExc2': 2.5 * ms,
    'refractoryPeriod': 1 * ms,  # overridden by the above but necessary
    'membraneCapacitanceExc': 200 * pF,  # dictated by surface area
    'membraneCapacitanceExc2': 240 * pF,  # so it's smaller for inhibitory neurons
    'gLeakExc': 10 * nS,  # in theory dictated by density of Na+/K+ pump, etc
    'gLeakExc2': 8 * nS,  # so it's the same between the two

    # ephys params

    'noiseSigma': 1 * mV,
    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    'eLeakInh': -60 * mV,
    'vResetInh': -60 * mV,
    'vThreshInh': -42 * mV,
    'betaAdaptInh': 2 * nA * ms,
    'refractoryPeriodInh': 1 * ms,
    'membraneCapacitanceInh': 120 * pF,  # so it's smaller for inhibitory neurons
    'gLeakInh': 8 * nS,  # so it's the same between the two

    # ephys params

    # synaptic params
    # 'jEE': 280 * mV,
    # 'jEI': 70 * mV,
    # 'jIE': 500 * mV,
    # 'jII': 100 * mV,

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

}

paramsJercogEphysBuono5 = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'simName': 'classicJercogEphysBuonoBen1',
    'paramSet': 'buonoEphysBen1',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 250 * ms,
    # 'updateMethod': 'exact',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'eLeakExc': -65 * mV,
    'eLeakExc2': -65 * mV,
    'vResetExc': -65 * mV,
    'vResetExc2': -65 * mV,
    'vThreshExc': -48 * mV,
    'vThreshExc2': -43 * mV,
    'adaptTau': 500 * ms,
    'betaAdaptExc': 10 * nA * ms,  # 14.5 * nA * ms,
    'betaAdaptExc2': 10 * nA * ms,  # 14.5 * nA * ms,
    'refractoryPeriodExc': 2.5 * ms,
    'refractoryPeriodExc2': 2.5 * ms,
    'refractoryPeriod': 1 * ms,  # overridden by the above but necessary
    'membraneCapacitanceExc': 200 * pF,  # dictated by surface area
    'membraneCapacitanceExc2': 240 * pF,  # so it's smaller for inhibitory neurons
    'gLeakExc': 10 * nS,  # in theory dictated by density of Na+/K+ pump, etc
    'gLeakExc2': 8 * nS,  # so it's the same between the two

    # ephys params

    'noiseSigma': 1 * mV,
    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    'eLeakInh': -65 * mV,
    'vResetInh': -65 * mV,
    'vThreshInh': -42 * mV,
    'betaAdaptInh': 2 * nA * ms,
    'refractoryPeriodInh': 1 * ms,
    'membraneCapacitanceInh': 120 * pF,  # so it's smaller for inhibitory neurons
    'gLeakInh': 8 * nS,  # so it's the same between the two

    # ephys params

    # synaptic params
    # 'jEE': 280 * mV,
    # 'jEI': 70 * mV,
    # 'jIE': 500 * mV,
    # 'jII': 100 * mV,

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

}

paramsJercogEphysBuono6 = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'simName': 'classicJercogEphysBuonoBen1',
    'paramSet': 'buonoEphysBen1',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 250 * ms,
    # 'updateMethod': 'exact',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'eLeakExc': -65 * mV,
    'eLeakExc2': -65 * mV,
    'vResetExc': -65 * mV,
    'vResetExc2': -65 * mV,
    'vThreshExc': -52 * mV,
    'vThreshExc2': -46 * mV,
    'adaptTau': 500 * ms,
    'betaAdaptExc': 10 * nA * ms,  # 14.5 * nA * ms,
    'betaAdaptExc2': 10 * nA * ms,  # 14.5 * nA * ms,
    'refractoryPeriodExc': 2.5 * ms,
    'refractoryPeriodExc2': 2.5 * ms,
    'refractoryPeriod': 1 * ms,  # overridden by the above but necessary
    'membraneCapacitanceExc': 200 * pF,  # dictated by surface area
    'membraneCapacitanceExc2': 240 * pF,  # so it's smaller for inhibitory neurons
    'gLeakExc': 10 * nS,  # in theory dictated by density of Na+/K+ pump, etc
    'gLeakExc2': 8 * nS,  # so it's the same between the two

    # ephys params

    'noiseSigma': 1 * mV,
    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    'eLeakInh': -65 * mV,
    'vResetInh': -65 * mV,
    'vThreshInh': -43 * mV,
    'betaAdaptInh': 1 * nA * ms,
    'refractoryPeriodInh': 1 * ms,
    'membraneCapacitanceInh': 120 * pF,  # so it's smaller for inhibitory neurons
    'gLeakInh': 8 * nS,  # so it's the same between the two

    # ephys params

    # synaptic params
    # 'jEE': 280 * mV,
    # 'jEI': 70 * mV,
    # 'jIE': 500 * mV,
    # 'jII': 100 * mV,

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

}

paramsJercogEphysBuonoBen1 = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'simName': 'classicJercogEphysBuonoBen1',
    'paramSet': 'buonoEphysBen1',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 250 * ms,
    # 'updateMethod': 'exact',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'eLeakExc': -60 * mV,
    'eLeakExc2': -60 * mV,
    'vResetExc': -60 * mV,
    'vResetExc2': -60 * mV,
    'vThreshExc': -52 * mV,
    'vThreshExc2': -42 * mV,
    'adaptTau': 500 * ms,
    'betaAdaptExc': 10 * nA * ms,
    'betaAdaptExc2': 10 * nA * ms,
    'refractoryPeriodExc': 2.5 * ms,
    'refractoryPeriodExc2': 2.5 * ms,
    'refractoryPeriod': 1 * ms,  # overridden by the above but necessary
    'membraneCapacitanceExc': 200 * pF,  # dictated by surface area
    'membraneCapacitanceExc2': 200 * pF,  # so it's smaller for inhibitory neurons
    'gLeakExc': 9 * nS,  # in theory dictated by density of Na+/K+ pump, etc
    'gLeakExc2': 6.5 * nS,  # so it's the same between the two

    # ephys params
    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    'eLeakInh': -60 * mV,
    'vResetInh': -60 * mV,
    'vThreshInh': -40 * mV,
    'betaAdaptInh': 0 * nA * ms,
    'refractoryPeriodInh': 1 * ms,
    'membraneCapacitanceInh': 100 * pF,  # so it's smaller for inhibitory neurons
    'gLeakInh': 10 * nS,  # so it's the same between the two

    # ephys params

    # synaptic params
    # 'jEE': 280 * mV,
    # 'jEI': 70 * mV,
    # 'jIE': 500 * mV,
    # 'jII': 100 * mV,
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

}

paramsJercogEphysBuonoBen11 = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'simName': 'classicJercogEphysBuonoBen1',
    'paramSet': 'buonoEphys',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 250 * ms,
    # 'updateMethod': 'exact',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'refractoryPeriod': 1 * ms,  # overridden but necessary
    'adaptTau': 500 * ms,

    'eLeakExc': -60 * mV,
    'vResetExc': -60 * mV,
    'vThreshExc': -52 * mV,
    'betaAdaptExc': 11 * nA * ms,
    'refractoryPeriodExc': 2.5 * ms,
    'membraneCapacitanceExc': 200 * pF,  # dictated by surface area
    'gLeakExc': 10 * nS,  # in theory dictated by density of Na+/K+ pump, etc

    'eLeakInh': -60 * mV,
    'vResetInh': -60 * mV,
    'vThreshInh': -42 * mV,
    'betaAdaptInh': 10 * nA * ms,
    'refractoryPeriodInh': 2.5 * ms,
    'membraneCapacitanceInh': 200 * pF,  # so it's smaller for inhibitory neurons
    'gLeakInh': 6.5 * nS,  # so it's the same between the two

    # 'eLeakInh': -60 * mV,
    # 'vResetInh': -60 * mV,
    # 'vThreshInh': -42 * mV,
    # 'betaAdaptInh': 2 * nA * ms,
    # 'refractoryPeriodInh': 1 * ms,
    # 'membraneCapacitanceInh': 120 * pF,  # so it's smaller for inhibitory neurons
    # 'gLeakInh': 8 * nS,  # so it's the same between the two

    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

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

}

paramsJercogEphysBuonoBen2 = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'simName': 'classicJercogEphysBuono',
    'paramSet': 'buonoEphysBen2',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 250 * ms,
    # 'updateMethod': 'exact',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'eLeakExc': -60 * mV,
    'eLeakExc2': -60 * mV,
    'vResetExc': -60 * mV,
    'vResetExc2': -60 * mV,
    'vThreshExc': -53 * mV,
    'vThreshExc2': -48 * mV,
    'adaptTau': 500 * ms,
    'betaAdaptExc': 10 * nA * ms,
    'betaAdaptExc2': 10 * nA * ms,
    'refractoryPeriodExc': 2.5 * ms,
    'refractoryPeriodExc2': 2.5 * ms,
    'refractoryPeriod': 1 * ms,  # overridden by the above but necessary
    'membraneCapacitanceExc': 220 * pF,  # dictated by surface area
    'membraneCapacitanceExc2': 300 * pF,  # so it's smaller for inhibitory neurons
    'gLeakExc': 10 * nS,  # in theory dictated by density of Na+/K+ pump, etc
    'gLeakExc2': 10 * nS,  # so it's the same between the two

    'eLeakInh': -60 * mV,
    'vResetInh': -60 * mV,
    'vThreshInh': -40 * mV,
    'betaAdaptInh': 0 * nA * ms,
    'refractoryPeriodInh': 1 * ms,
    'membraneCapacitanceInh': 100 * pF,  # so it's smaller for inhibitory neurons
    'gLeakInh': 10 * nS,  # so it's the same between the two

    # ephys params
    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    # synaptic params
    # 'jEE': 280 * mV,
    # 'jEI': 70 * mV,
    # 'jIE': 500 * mV,
    # 'jII': 100 * mV,
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

}

paramsJercogEphysBuonoBen21 = {
    # save / figs
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'simName': 'classicJercogEphysBuono',
    'paramSet': 'buonoEphysBen2',
    'saveWithDate': True,

    # global sim params
    'dt': 0.1 * ms,
    'duration': 250 * ms,
    # 'updateMethod': 'exact',
    'reportType': 'stdout',
    'reportPeriod': 10 * second,
    'doProfile': False,

    # recording parameters
    'propSpikemon': 1,
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'refractoryPeriod': 1 * ms,  # overridden but necessary
    'adaptTau': 500 * ms,

    'eLeakExc': -60 * mV,
    'vResetExc': -60 * mV,
    'vThreshExc': -50 * mV,
    'betaAdaptExc': 11 * nA * ms,
    'refractoryPeriodExc': 2.5 * ms,
    'membraneCapacitanceExc': 200 * pF,  # dictated by surface area
    'gLeakExc': 10 * nS,  # in theory dictated by density of Na+/K+ pump, etc

    'eLeakInh': -60 * mV,
    'vResetInh': -60 * mV,
    'vThreshInh': -45 * mV,
    'betaAdaptInh': 11 * nA * ms,
    'refractoryPeriodInh': 2.5 * ms,
    'membraneCapacitanceInh': 280 * pF,  # so it's smaller for inhibitory neurons
    'gLeakInh': 10 * nS,  # so it's the same between the two

    # 'eLeakInh': -60 * mV,
    # 'vResetInh': -60 * mV,
    # 'vThreshInh': -42 * mV,
    # 'betaAdaptInh': 2 * nA * ms,
    # 'refractoryPeriodInh': 1 * ms,
    # 'membraneCapacitanceInh': 120 * pF,  # so it's smaller for inhibitory neurons
    # 'gLeakInh': 8 * nS,  # so it's the same between the two

    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    # synaptic params
    # 'jEE': 280 * mV,
    # 'jEI': 70 * mV,
    # 'jIE': 500 * mV,
    # 'jII': 100 * mV,
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

}

paramsDestexhe = {
    # save / figs?
    'saveFolder': 'C:/Users/mikejseay/Dropbox/UCLA/courses/covid-era-modeling/figs/',
    'simName': 'destexhe',
    'paramSet': 'classicDestexhe',
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
    'recordStateVariables': ['v', ],  # ['v', 'w', 'ge', 'gi'],
    'indsRecordStateExc': [0, ],
    'indsRecordStateInh': [0, ],

    # network params
    'nUnits': 1e4,  # ***
    'propInh': 0.2,
    'propConnect': 0.05,  # ***
    'propKicked': 0.1,
    #     'propPoissonInputs': 0.05,
    'poissonInputRate': 0.315 * Hz,  # ***
    'poissonInputWeightMultiplier': 2,  # ***

    # external inputs
    'kickMode': 'fixed',  # fixed or Poisson
    'kickTau': 0.5 * ms,
    'kickDur': 2 * ms,
    'kickLambda': 0.4 * second,
    'kickMinimumISI': 1 * second,
    'kickMaximumISI': 10 * second,
    'kickTimes': [0.1, 0.8] * second,
    'kickSizes': [1, 1],
    'kickAmplitudeExc': 1.5 * nA,
    'kickAmplitudeInh': 1.5 * nA,

    # unit params
    'membraneCapacitance': 200 * pF,  # 150 or 200
    'gLeak': 10 * nS,
    'eLeakExc': -63 * mV,  # -65 or -63
    'eLeakInh': -65 * mV,
    'vThresh': -50 * mV,  # default is 50!!
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
    # make NMDA channels!
    # a gated conductance
    # time constant is much higher (e.g. 50 ms)
    # 50% mark of sigmoid is maybe 50 mV
    'eExcSyn': 0 * mV,
    'eInhSyn': -80 * mV,

}

paramsDestexheEphysOrig = {
    'simName': 'destexheEphysOrig',
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'paramSet': 'destexheEphys',
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
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'eLeakExc': -63 * mV,  # -65 or -63
    'eLeakInh': -65 * mV,
    'vThreshExc': -50 * mV,  # default is 50!! true thresh is vThresh + 5 * delta (Exc = -40, Inh = -47.5)
    'vThreshInh': -50 * mV,
    'deltaVExc': 2 * mV,
    'deltaVInh': 0.5 * mV,
    'aExc': 0 * nS,  # 0 or 4 * ns
    'aInh': 0 * nS,
    'bExc': 40 * pA,  # 20-60 pA
    'bInh': 0 * pA,
    'adaptTau': 500 * ms,
    'membraneCapacitanceExc': 200 * pF,  # 150 or 200
    'membraneCapacitanceInh': 200 * pF,  # 150 or 200
    'gLeakExc': 10 * nS,
    'gLeakInh': 10 * nS,
    'vResetExc': -63 * mV,
    'vResetInh': -65 * mV,
    'refractoryPeriodExc': 5 * ms,
    'refractoryPeriodInh': 5 * ms,

    # ephys params
    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    # synaptic params
    'qExtExc': 0.6 * uS,
    'qExc': 0.6 * uS,  # *** # 1.5 * nS * 1e4 * (1 - .2) * 0.05,
    'qInh': 0.5 * uS,  # *** # 5 * nS * 1e4 * .2 * 0.05,
    'tauSynExc': 5 * ms,
    'tauSynInh': 5 * ms,
    'eExcSyn': 0 * mV,
    'eInhSyn': -80 * mV,

}

paramsDestexheEphysBuono = {
    'simName': 'destexheEphysBuono',
    'saveFolder': 'C:/Users/mikejseay/Documents/BrianResults/ephys/',
    'paramSet': 'buonoEphys',
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
    'nUnits': 2,  # ***
    'propInh': 0.5,

    # unit params
    'eLeakExc': -63 * mV,  # -65 or -63
    'eLeakInh': -65 * mV,
    'vThreshExc': -53 * mV,  # default is 50!! true thresh is vThresh + 5 * delta (Exc = -40, Inh = -47.5)
    'vThreshInh': -40 * mV,
    'deltaVExc': 2 * mV,
    'deltaVInh': 0.5 * mV,
    'aExc': 0 * nS,  # 0 or 4 * ns
    'aInh': 0 * nS,
    'bExc': 20 * pA,  # 20-60 pA
    'bInh': 0 * pA,
    'adaptTau': 500 * ms,
    'membraneCapacitanceExc': 200 * pF,  # 150 or 200
    'membraneCapacitanceInh': 80 * pF,  # 150 or 200
    'gLeakExc': 10 * nS,
    'gLeakInh': 8 * nS,
    'vResetExc': -63 * mV,
    'vResetInh': -52.5 * mV,
    'refractoryPeriodExc': 2.5 * ms,
    'refractoryPeriodInh': 1 * ms,

    # ephys params
    'iExtRange': linspace(0, .3, 31) * nA,
    'iDur': 250 * ms,

    # synaptic params
    'qExtExc': 0.6 * uS,
    'qExc': 0.6 * uS,  # *** # 1.5 * nS * 1e4 * (1 - .2) * 1,
    'qInh': 0.5 * uS,  # *** # 5 * nS * 1e4 * .2 * 0.05,
    'tauSynExc': 5 * ms,
    'tauSynInh': 5 * ms,
    'eExcSyn': 0 * mV,
    'eInhSyn': -80 * mV,

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
