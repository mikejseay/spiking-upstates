import sys
import numpy as np
from brian2 import defaultclock, ms, pA, nA, Hz, seed, mV

from params import paramsJercog as p
from params import (paramsJercogEphysBuono22, paramsJercogEphysBuono4, paramsJercogEphysBuono5, paramsJercogEphysBuono6,
                    paramsJercogEphysBuono7, paramsJercogEphysBuono7InfUp)
from generate import convert_kicks_to_current_series
from trainer import JercogTrainer
from results import Results

p['useNewEphysParams'] = False
p['useSecondPopExc'] = False
ephysParams = paramsJercogEphysBuono7.copy()

if p['useNewEphysParams']:
    # remove protected keys from the dict whose params are being imported
    protectedKeys = ('nUnits', 'propInh', 'duration')
    for pK in protectedKeys:
        del ephysParams[pK]
    p.update(ephysParams)

defaultclock.dt = p['dt']

if sys.platform == 'win32':
    p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
elif sys.platform == 'linux':
    p['saveFolder'] = '/u/home/m/mikeseay/BrianResults/'
p['saveWithDate'] = True
p['useOldWeightMagnitude'] = True
p['disableWeightScaling'] = True
p['applyLogToFR'] = False
p['setMinimumBasedOnBalance'] = False
p['recordMovieVariables'] = True
p['downSampleVoltageTo'] = 1 * ms
p['dtHistPSTH'] = 10 * ms

# simulation params
p['nUnits'] = 2e3
p['propConnect'] = 0.25
# p['noiseSigma'] = 2.5 * mV  # 2.5 * mV

# define parameters
p['setUpFRExc'] = 5 * Hz
p['setUpFRInh'] = 14 * Hz
p['tauUpFRTrials'] = 2
p['useRule'] = 'cross-homeo-pre-scalar'  # cross-homeo or balance
rngSeed = 8
p['allowAutapses'] = False
p['nameSuffix'] = 'explodeDealTest'
# cross-homeo-scalar and cross-homeo-scalar-homeo are the new ones
p['saveTermsSeparately'] = False
# defaultEqual, defaultNormal, defaultNormalScaled, defaultUniform,
# randomUniform, randomUniformMid, randomUniformLow, randomUniformSaray, randomUniformSarayMid, randomUniformSarayHigh

# turn off adaptation?
# p['betaAdaptExc'] = 0 * nA * ms
# p['betaAdaptExc2'] = 0 * nA * ms
# p['betaAdaptInh'] = 0 * nA * ms

p['initWeightMethod'] = 'seed' + str(rngSeed)
# p['initWeightMethod'] = 'goodCrossHomeoExamp'
# p['initWeightMethod'] = 'goodCrossHomeoExampBuono'
# p['initWeightMethod'] = 'guessLowWeights2e3p025LogNormal2'
# p['initWeightMethod'] = 'guessBuono2Weights2e3p025LogNormal2'
# p['initWeightMethod'] = 'guessGoodWeights2e3p1LogNormal'
# p['initWeightMethod'] = 'guessGoodWeights2e3p025'
# p['initWeightMethod'] = 'guessGoodWeights2e3p025Normal'
# p['initWeightMethod'] = 'guessGoodWeights2e3p025LogNormal'
# p['initWeightMethod'] = 'guessBuono6Weights2e3p025Beta10'
# p['initWeightMethod'] = 'guessBuono7Weights2e3p025SlightLow'
# p['initWeightMethod'] = 'guessBuono4Weights2e3p025LogNormal'
# p['initWeightMethod'] = 'randomUniformSarayHigher'
# p['initWeightMethod'] = 'guessZeroActivityWeights2e3p025LogNormal'
# p['initWeightMethod'] = 'guessHighActivityWeights2e3p025LogNormal'
# p['initWeightMethod'] = 'guessUpperLeftWeights2e3p025LogNormal'
# p['initWeightMethod'] = 'guessLowerRightWeights2e3p025LogNormal'
# p['initWeightMethod'] = 'guessZeroActivityWeights2e3p025'
# # p['initWeightMethod'] = 'guessLowActivityWeights2e3p025'
# p['initWeightMethod'] = 'randomUniformSarayHigh5e3p02Converge'
# p['initWeightMethod'] = 'randomUniformSarayHigh'
# p['initWeightMethod'] = 'randomUniformMidUnequal'

# p['initWeightMethod'] = 'resumePrior'  # note this completely overwrites ALL values of the p parameter
# p['saveFolder'] += 'cross-homeo-pre-outer-homeo/'
# p['initWeightPrior'] = 'classicJercog_2000_0p25_cross-homeo-pre-outer-homeo_resumePrior_seed2_2021-09-03-10-23_results'

# p['initWeightMethod'] = 'resumePrior'
# p['initWeightPrior'] = 'buonoEphysBen1_2000_0p25_cross-homeo-pre-outer-homeo_guessBuono7Weights2e3p025SlightLow__2021-09-04-08-20_results'

p['kickType'] = 'spike'  # kick or spike
p['jEEScaleRatio'] = None
p['jIEScaleRatio'] = None
p['jEIScaleRatio'] = None
p['jIIScaleRatio'] = None

p['maxAllowedFRExc'] = 2 * p['setUpFRExc'] / Hz
p['maxAllowedFRInh'] = 2 * p['setUpFRInh'] / Hz

p['nTrials'] = 3000  # 6765
# p['saveTrials'] = [1, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]  # 1-indexed
# p['saveTrials'] = [1, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]  # 1-indexed
p['saveTrials'] = np.arange(0, p['nTrials'], 100)

p['nUnitsToSpike'] = int(np.round(0.05 * p['nUnits']))
p['timeToSpike'] = 100 * ms
p['timeAfterSpiked'] = 1400 * ms
if p['useNewEphysParams']:
    p['spikeInputAmplitude'] = 0.96  # nA
else:
    p['spikeInputAmplitude'] = 0.98  # 0.95  # 1.34  # 1.03  # nA

if p['useRule'][:5] == 'cross' or p['useRule'] == 'homeo':
    p['alpha1'] = 0.001 * pA / Hz
    p['alpha2'] = None
    p['tauPlasticityTrials'] = None
    p['alphaBalance'] = None
    p['minAllowedWEE'] = 5 * pA
    p['minAllowedWEI'] = 5 * pA
    p['minAllowedWIE'] = 5 * pA
    p['minAllowedWII'] = 5 * pA
    p['maxAllowedWEE'] = 750 * pA
    p['maxAllowedWEI'] = 750 * pA
    p['maxAllowedWIE'] = 750 * pA
    p['maxAllowedWII'] = 750 * pA
elif p['useRule'][:7] == 'balance':
    # monolithic change version
    # p['alpha1'] = 0.05 * pA * pA / Hz / Hz / Hz / p['propConnect']
    # p['alpha2'] = 0.0005 * pA * pA / Hz / Hz / Hz / p['propConnect']
    # customized change version - no longer multiply by gain (Hz/amp) so must do that here
    p['alpha1'] = 0.01 * pA / p['propConnect']
    p['alpha2'] = 0.001 * pA / p['propConnect']
    p['tauPlasticityTrials'] = 100
    p['alphaBalance'] = 1 / p['tauPlasticityTrials']

    p['minAllowedWEE'] = 0.1 * pA / p['propConnect']
    p['minAllowedWEI'] = 0.1 * pA / p['propConnect']
    p['minAllowedWIE'] = 0.1 * pA / p['propConnect']
    p['minAllowedWII'] = 0.1 * pA / p['propConnect']

    # p['minAllowedWEE'] = 0.6 * pA / p['propConnect']  # * 800 this is empirical!!!
    # p['minAllowedWIE'] = 1.5 * pA / p['propConnect']  # * 800 this is empirical!!!
    # p['minAllowedWEI'] = 0.15 * pA / p['propConnect']  # * 200
    # p['minAllowedWII'] = 0.15 * pA / p['propConnect']  # * 200

if p['useRule'] == 'cross-homeo-pre-scalar-homeo-norm':
    p['alpha1'] *= 100

if p['useRule'][:18] == 'cross-homeo-scalar':
    p['alpha1'] *= 10

# params not important unless using "kick" instead of "spike"
p['propKicked'] = 0.1
p['duration'] = 1500 * ms
p['onlyKickExc'] = True
p['kickTimes'] = [100 * ms]
p['kickSizes'] = [1]
iKickRecorded = convert_kicks_to_current_series(p['kickDur'], p['kickTau'],
                                                p['kickTimes'], p['kickSizes'], p['duration'], p['dt'])
p['iKickRecorded'] = iKickRecorded

# ephys measurement params
p['baselineDur'] = 100 * ms
p['iDur'] = 250 * ms
p['afterDur'] = 100 * ms
p['iExtRange'] = np.linspace(0, .3, 301) * nA


# boring params
p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])
indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)
p['saveTrials'] = np.array(p['saveTrials']) - 1

# END OF PARAMS

if p['initWeightMethod'] == 'resumePrior':
    PR = Results()
    PR.init_from_file(p['initWeightPrior'], p['saveFolder'])
    p = dict(list(p.items()) + list(PR.p.items()))
    # p = PR.p.copy()  # note this completely overwrites all settings above
    # p['betaAdaptExc'] = 3 * nA * ms  # i apologize for this horrible hack
    p['nameSuffix'] = p['initWeightMethod'] + p['nameSuffix']  # a way to remember what it was...
    if 'seed' in p['nameSuffix']:  # this will only work for single digit seeds...
        rngSeed = int(p['nameSuffix'][p['nameSuffix'].find('seed') + 4])
    p['initWeightMethod'] = 'resumePrior'  # and then we put this back...
    if sys.platform == 'win32':
        p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
    elif sys.platform == 'linux':
        p['saveFolder'] = '/u/home/m/mikeseay/BrianResults/'
    JT = JercogTrainer(p)
    JT.set_up_network(priorResults=PR)  # additional argument allows synapses to be initialized from previous results
    # set RNG seeds...
    p['rngSeed'] = rngSeed
    rng = np.random.default_rng(rngSeed)  # for numpy
    seed(rngSeed)  # for Brian... will insert code to set the random number generator seed into the generated code
    p['rng'] = rng
else:
    JT = JercogTrainer(p)
    JT.calculate_unit_thresh_and_gain()
    # set RNG seeds...
    p['rngSeed'] = rngSeed
    rng = np.random.default_rng(rngSeed)  # for numpy
    seed(rngSeed)  # for Brian... will insert code to set the random number generator seed into the generated code
    p['rng'] = rng
    JT.set_up_network()

if p['useRule'] == 'balance' and p['setMinimumBasedOnBalance']:  # multiply by Hz below because set points lack units
    p['minAllowedWEE'] = (p['setUpFRInh'] / p['setUpFRExc'] * \
                         p['minAllowedWEI'] + (1 / p['gainExc'] + p['threshExc'] / p['setUpFRExc']) * Hz)
    p['minAllowedWIE'] = ((p['minAllowedWII'] + 1 / p['gainInh'] * Hz) * \
                         p['setUpFRInh'] / p['setUpFRExc'] + p['threshInh'] / p['setUpFRExc'] * Hz)
    # in Multi-Unit mins: wEE: 2.2 --> 0.030, WIE: 6 --> 0.075, WEI: 0.1 --> 0.005, WII: 0.1 --> 0.005
    # good weights in terms of sum(pre):  wEE: 5, WIE: 8.5, WEI: 1.3, WII: 1
    # therefore starting idea of good minimums for me would be
    # my good weights: wEE

JT.initalize_history_variables()
JT.initialize_weight_matrices()
JT.run()
JT.save_params()
JT.save_results()
