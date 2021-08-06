from brian2 import defaultclock, ms, pA, nA, Hz, seed
from params import paramsJercog as p
from params import paramsJercogEphysBuono
import numpy as np
from generate import convert_kicks_to_current_series
from trainer import JercogTrainer
from results import Results


defaultclock.dt = p['dt']

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
# p['saveFolder'] = '/u/home/m/mikeseay/BrianResults/'
p['saveWithDate'] = True
p['useOldWeightMagnitude'] = True
p['disableWeightScaling'] = True
p['useNewEphysParams'] = False
p['applyLogToFR'] = False
p['setMinimumBasedOnBalance'] = False
p['recordMovieVariables'] = True
p['downSampleVoltageTo'] = 1 * ms
ephysParams = paramsJercogEphysBuono.copy()

# simulation params
p['nUnits'] = 2e3
p['propConnect'] = 0.25

# define parameters
p['setUpFRExc'] = 5 * Hz
p['setUpFRInh'] = 14 * Hz
p['tauUpFRTrials'] = 1
p['useRule'] = 'cross-homeo-pre-scalar-homeo'  # cross-homeo or balance
rngSeed = None
p['nameSuffix'] = 'movAvg1'
# cross-homeo-scalar and cross-homeo-scalar-homeo are the new ones
p['saveTermsSeparately'] = True
# defaultEqual, defaultNormal, defaultNormalScaled, defaultUniform,
# randomUniform, randomUniformMid, randomUniformLow, randomUniformSaray, randomUniformSarayMid, randomUniformSarayHigh

# p['initWeightMethod'] = 'seed' + str(rngSeed)
# p['initWeightMethod'] = 'guessGoodWeights2e3p025LogNormal'
# p['initWeightMethod'] = 'guessZeroActivityWeights2e3p025LogNormal'
# p['initWeightMethod'] = 'guessHighActivityWeights2e3p025LogNormal'
p['initWeightMethod'] = 'guessUpperLeftWeights2e3p025LogNormal'
# p['initWeightMethod'] = 'guessLowerRightWeights2e3p025LogNormal'
# p['initWeightMethod'] = 'guessZeroActivityWeights2e3p025'
# # p['initWeightMethod'] = 'guessLowActivityWeights2e3p025'
# p['initWeightMethod'] = 'randomUniformSarayHigh5e3p02Converge'
# p['initWeightMethod'] = 'randomUniformSarayHigh'
# p['initWeightMethod'] = 'randomUniformMidUnequal'
# p['initWeightMethod'] = 'resumePrior'  # note this completely overwrites ALL values of the p parameter
# p['initWeightPrior'] = 'classicJercog_2000_0p25_cross-homeo-pre-scalar-homeo_seed8__2021-08-02-14-33_results'

p['kickType'] = 'spike'  # kick or spike
p['jEEScaleRatio'] = None
p['jIEScaleRatio'] = None
p['jEIScaleRatio'] = None
p['jIIScaleRatio'] = None

p['maxAllowedFRExc'] = 2 * p['setUpFRExc'] / Hz
p['maxAllowedFRInh'] = 2 * p['setUpFRInh'] / Hz

p['nTrials'] = 6765  # 6765
# p['saveTrials'] = [1, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]  # 1-indexed
# p['saveTrials'] = [1, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]  # 1-indexed
p['saveTrials'] = [1, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]  # 1-indexed

p['nUnitsToSpike'] = int(np.round(0.05 * p['nUnits']))
p['timeToSpike'] = 100 * ms
p['timeAfterSpiked'] = 1400 * ms
p['spikeInputAmplitude'] = 0.98 * nA

if p['useRule'][:5] == 'cross' or p['useRule'] == 'homeo':
    p['alpha1'] = 0.002 * pA / Hz / p['propConnect']  # 0.005
    p['alpha2'] = None
    p['tauPlasticityTrials'] = None
    p['alphaBalance'] = None
    p['minAllowedWEE'] = 0.1 * pA / p['propConnect']
    p['minAllowedWEI'] = 0.1 * pA / p['propConnect']
    p['minAllowedWIE'] = 0.1 * pA / p['propConnect']
    p['minAllowedWII'] = 0.1 * pA / p['propConnect']
elif p['useRule'][:7] == 'balance':
    # monolithic change version
    # p['alpha1'] = 0.05 * pA * pA / Hz / Hz / Hz / p['propConnect']
    # p['alpha2'] = 0.0005 * pA * pA / Hz / Hz / Hz / p['propConnect']
    # customized change version - no longer multiply by gain (Hz/amp) so must do that here
    p['alpha1'] = 0.05 * pA / p['propConnect']
    p['alpha2'] = 0.005 * pA / p['propConnect']
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

# boring params
p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])
indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)
p['saveTrials'] = np.array(p['saveTrials']) - 1

if p['useNewEphysParams']:
    # remove protected keys from the dict whose params are being imported
    protectedKeys = ('nUnits', 'propInh', 'duration')
    for pK in protectedKeys:
        del ephysParams[pK]
    p.update(ephysParams)

# END OF PARAMS

# set RNG seeds...
p['rngSeed'] = rngSeed
rng = np.random.default_rng(rngSeed)  # for numpy
seed(rngSeed)  # for Brian... will insert code to set the random number generator seed into the generated code
p['rng'] = rng

if p['initWeightMethod'] == 'resumePrior':
    PR = Results()
    PR.init_from_file(p['initWeightPrior'], p['saveFolder'])
    p = PR.p.copy()  # note this completely overwrites all settings above
    p['nameSuffix'] = p['initWeightMethod'] + p['nameSuffix']  # a way to remember what it was...
    p['initWeightMethod'] = 'resumePrior'  # and then we put this back...
    JT = JercogTrainer(p)
    JT.set_up_network(priorResults=PR)  # additional argument allows synapses to be initialized from previous results
else:
    JT = JercogTrainer(p)
    JT.calculate_unit_thresh_and_gain()
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
