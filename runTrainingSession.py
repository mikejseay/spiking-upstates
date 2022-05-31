import os
import numpy as np
from brian2 import defaultclock, ms, pA, Hz, seed

# from params import paramsJercog as p
from params import paramsJercogEphysBuono as p
from trainer import JercogTrainer

defaultclock.dt = p['dt']

p['saveFolder'] = os.path.join(os.getcwd(), 'results')
p['saveWithDate'] = True
p['recordMovieVariables'] = True
p['downSampleVoltageTo'] = 1 * ms
p['dtHistPSTH'] = 10 * ms

# simulation params
p['nUnits'] = 2e3
p['propInh'] = 0.2
p['propConnect'] = 0.25

# define parameters
p['setUpFRExc'] = 5 * Hz
p['setUpFRInh'] = 14 * Hz
p['tauUpFRTrials'] = 2
p['useRule'] = 'cross-homeo-pre-outer-homeo'
rngSeed = None
p['allowAutapses'] = False
p['nameSuffix'] = 'buonoParams'
p['saveTermsSeparately'] = False
p['initWeightMethod'] = 'goodCrossHomeoExamp'

p['kickType'] = 'spike'  # kick or spike

p['nTrials'] = 8000
p['saveTrials'] = np.arange(0, p['nTrials'], 100)

p['nUnitsToSpike'] = int(np.round(0.05 * p['nUnits']))
p['timeToSpike'] = 100 * ms
p['timeAfterSpiked'] = 1400 * ms
p['spikeInputAmplitude'] = 0.98

p['alpha1'] = 0.005 * pA / Hz
p['alpha2'] = None
p['tauPlasticityTrials'] = None
p['alphaBalance'] = None
p['minAllowedWEE'] = 5 * pA
p['minAllowedWEI'] = 5 * pA
p['minAllowedWIE'] = 5 * pA
p['minAllowedWII'] = 5 * pA
p['maxAllowedWEE'] = 750 * pA
p['maxAllowedWIE'] = 750 * pA
p['maxAllowedWEI'] = 750 * pA
p['maxAllowedWII'] = 750 * pA

# boring params
p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])
indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)
p['saveTrials'] = np.array(p['saveTrials']) - 1

# END OF PARAMS

JT = JercogTrainer(p)

# set RNG seeds...
p['rngSeed'] = rngSeed
rng = np.random.default_rng(rngSeed)  # for numpy
seed(rngSeed)  # for Brian... will insert code to set the random number generator seed into the generated code
p['rng'] = rng
JT.set_up_network()

JT.initalize_history_variables()
JT.initialize_weight_matrices()
JT.run()
JT.save_params()
JT.save_results()
