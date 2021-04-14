from brian2 import *
from params import paramsDestexhe, paramsDestexheEphysBuono, paramsDestexheEphysOrig
from network import DestexheNetwork

p = paramsDestexhe.copy()
USE_NEW_EPHYS_PARAMS = True

# remove protected keys from the dict whose params are being imported
# ephysParams = paramsDestexheEphysOrig.copy()
ephysParams = paramsDestexheEphysBuono.copy()
protectedKeys = ('nUnits', 'propInh', 'duration')
for pK in protectedKeys:
    del ephysParams[pK]

if USE_NEW_EPHYS_PARAMS:
    p.update(ephysParams)

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

defaultclock.dt = p['dt']

# determine 'up crit' empirically

p['simName'] = 'destexheUpCrit'
p['nUnits'] = 1e3
p['propConnect'] = 0.1  # 0.05

SEPARATE_BY_BUNDLE = True

# NOTE: these represent the TOTAL synaptic input a neuron in the post-class from all presynaptic units in the pre-class
# (i.e. each connection will actually be qEE / n_incoming)
p['qEE'] = 0.6 * uS
p['qIE'] = 0.6 * uS
p['qEI'] = 0.5 * uS
p['qII'] = 0.5 * uS

p['pEE'] = 0.1
p['pIE'] = 0.5
p['pEI'] = 0.5
p['pII'] = 0.5

USE_DISTRIBUTED_WEIGHTS = False
normalMean = 1
normalSD = 0.2

DN = DestexheNetwork(p)
DN.initialize_network()
DN.initialize_units()

if USE_DISTRIBUTED_WEIGHTS:
    DN.initialize_recurrent_synapses_4bundles_distributed(normalMean=normalMean, normalSD=normalSD)
else:
    if SEPARATE_BY_BUNDLE:
        DN.initialize_recurrent_synapses_4bundles_separate()
    else:
        DN.initialize_recurrent_synapses_4bundles()

DN.prepare_upCrit_experiment(minUnits=190, maxUnits=190, unitSpacing=20, timeSpacing=1000 * ms)

DN.create_monitors()
DN.run()
DN.save_results()
DN.save_params()
