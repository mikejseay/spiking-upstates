from brian2 import *
from params import paramsDestexhe as p
from network import DestexheNetwork

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

defaultclock.dt = p['dt']

# determine 'up crit' empirically

p['simName'] = 'destexheUpCritFullConn'
p['nUnits'] = 1e3
p['propConnect'] = 1

USE_DISTRIBUTED_WEIGHTS = True
normalMean = 1
normalSD = 0.2

DN = DestexheNetwork(p)
DN.initialize_network()
DN.initialize_units()

if USE_DISTRIBUTED_WEIGHTS:
    DN.initialize_recurrent_synapses_4bundles_distributed(normalMean=normalMean, normalSD=normalSD)
else:
    DN.initialize_recurrent_synapses_4bundles()

DN.prepare_upCrit_experiment(minUnits=40, maxUnits=80, unitSpacing=10, timeSpacing=3000 * ms)

DN.create_monitors()
DN.run()
DN.save_results()
DN.save_params()
